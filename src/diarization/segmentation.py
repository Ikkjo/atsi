"""Diarization-specific speech segment preparation.

The shared VAD component returns speech regions. This module turns those
regions into short clips suitable for speaker embedding extraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.diarization.vad import PyannoteVAD, SpeechRegion, normalize_speech_regions


EmbeddingSegment = dict[str, float | int | str | None]


@dataclass(frozen=True)
class EmbeddingSegmentationConfig:
    """Configuration for embedding clip generation from VAD speech regions."""

    clip_duration_s: float = 1.5
    min_duration_s: float = 0.5
    merge_short_tail: bool = True


class DiarizationSegmenter:
    """Prepare speaker-embedding clips from shared pyannote VAD output."""

    def __init__(
        self,
        vad: PyannoteVAD | None = None,
        config: EmbeddingSegmentationConfig | None = None,
    ) -> None:
        self.vad = vad or PyannoteVAD()
        self.config = config or EmbeddingSegmentationConfig()

    def detect_segments(self, audio_path: str | Path) -> list[EmbeddingSegment]:
        """Run shared VAD and split detected speech into embedding clips."""
        speech_regions = self.vad.detect_speech(audio_path)
        return prepare_embedding_segments(speech_regions, config=self.config)


def prepare_embedding_segments(
    speech_regions: list[dict[str, Any]],
    config: EmbeddingSegmentationConfig | None = None,
) -> list[EmbeddingSegment]:
    """Split VAD regions into fixed-duration clips for ECAPA extraction.

    Very short tails are merged into the previous clip from the same VAD region
    when possible. Standalone regions shorter than ``min_duration_s`` are
    discarded because they usually produce unstable speaker embeddings.
    """
    cfg = config or EmbeddingSegmentationConfig()
    _validate_config(cfg)

    segments: list[EmbeddingSegment] = []
    for region_index, region in enumerate(normalize_speech_regions(speech_regions)):
        region_segments = _split_region(region, region_index, cfg)
        for segment in region_segments:
            segment["segment_id"] = f"seg_{len(segments):06d}"
            segments.append(segment)
    return segments


def _split_region(
    region: SpeechRegion,
    region_index: int,
    config: EmbeddingSegmentationConfig,
) -> list[EmbeddingSegment]:
    start = float(region["start"])
    end = float(region["end"])
    duration = end - start
    if duration < config.min_duration_s:
        return []

    region_segments: list[EmbeddingSegment] = []
    cursor = start
    while end - cursor > config.clip_duration_s:
        chunk_end = cursor + config.clip_duration_s
        region_segments.append(
            _segment_dict(cursor, chunk_end, region_index, region.get("score"))
        )
        cursor = chunk_end

    tail_duration = end - cursor
    if tail_duration >= config.min_duration_s:
        region_segments.append(_segment_dict(cursor, end, region_index, region.get("score")))
    elif config.merge_short_tail and region_segments:
        region_segments[-1]["end"] = end
        region_segments[-1]["duration"] = float(region_segments[-1]["end"]) - float(
            region_segments[-1]["start"]
        )
    elif not region_segments and tail_duration >= config.min_duration_s:
        region_segments.append(_segment_dict(cursor, end, region_index, region.get("score")))

    return region_segments


def _segment_dict(
    start: float,
    end: float,
    source_region_index: int,
    score: float | None,
) -> EmbeddingSegment:
    return {
        "segment_id": None,
        "start": float(start),
        "end": float(end),
        "duration": float(end - start),
        "source_region_index": int(source_region_index),
        "score": float(score) if score is not None else None,
    }


def _validate_config(config: EmbeddingSegmentationConfig) -> None:
    if config.clip_duration_s <= 0:
        raise ValueError("clip_duration_s must be positive")
    if config.min_duration_s <= 0:
        raise ValueError("min_duration_s must be positive")
    if config.min_duration_s > config.clip_duration_s:
        raise ValueError("min_duration_s must be <= clip_duration_s")
