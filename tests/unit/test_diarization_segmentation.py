import pytest

from src.diarization.segmentation import (
    DiarizationSegmenter,
    EmbeddingSegmentationConfig,
    prepare_embedding_segments,
)


def test_prepare_embedding_segments_splits_to_embedding_clips():
    segments = prepare_embedding_segments(
        [{"start": 0.0, "end": 3.2, "score": 0.9}],
        config=EmbeddingSegmentationConfig(clip_duration_s=1.5, min_duration_s=0.5),
    )

    assert segments == [
        {
            "segment_id": "seg_000000",
            "start": 0.0,
            "end": 1.5,
            "duration": 1.5,
            "source_region_index": 0,
            "score": 0.9,
        },
        {
            "segment_id": "seg_000001",
            "start": 1.5,
            "end": 3.2,
            "duration": 1.7000000000000002,
            "source_region_index": 0,
            "score": 0.9,
        },
    ]


def test_prepare_embedding_segments_drops_too_short_regions():
    assert prepare_embedding_segments(
        [{"start": 0.0, "end": 0.3}],
        config=EmbeddingSegmentationConfig(clip_duration_s=1.5, min_duration_s=0.5),
    ) == []


def test_prepare_embedding_segments_keeps_valid_tail():
    segments = prepare_embedding_segments(
        [{"start": 10.0, "end": 12.1}],
        config=EmbeddingSegmentationConfig(clip_duration_s=1.5, min_duration_s=0.5),
    )

    assert [(segment["start"], segment["end"]) for segment in segments] == [
        (10.0, 11.5),
        (11.5, 12.1),
    ]


def test_prepare_embedding_segments_rejects_invalid_config():
    with pytest.raises(ValueError):
        prepare_embedding_segments(
            [{"start": 0.0, "end": 1.0}],
            config=EmbeddingSegmentationConfig(clip_duration_s=1.0, min_duration_s=2.0),
        )


def test_diarization_segmenter_uses_shared_vad_output():
    class FakeVAD:
        def detect_speech(self, audio_path):
            assert str(audio_path) == "meeting.wav"
            return [{"start": 0.0, "end": 1.0}]

    segmenter = DiarizationSegmenter(
        vad=FakeVAD(),
        config=EmbeddingSegmentationConfig(clip_duration_s=1.5, min_duration_s=0.5),
    )

    assert segmenter.detect_segments("meeting.wav")[0]["segment_id"] == "seg_000000"
