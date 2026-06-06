"""Shared pyannote VAD segmentation utilities.

This module is intentionally owned by diarization so ASR can reuse the same
speech regions without creating a second VAD pipeline.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.utils.hardware import get_device

logger = logging.getLogger(__name__)


SpeechRegion = dict[str, float | None]


@dataclass(frozen=True)
class PyannoteVADConfig:
    """Configuration for the shared pyannote VAD pipeline."""

    model_id: str = "pyannote/voice-activity-detection"
    auth_token: str | None = None
    min_duration_on: float | None = None
    min_duration_off: float | None = 0.3


class PyannoteVAD:
    """Lazy-loading wrapper around pyannote.audio voice activity detection."""

    def __init__(self, config: PyannoteVADConfig | None = None) -> None:
        self.config = config or PyannoteVADConfig()
        self.device = get_device()
        self._pipeline = None

    @property
    def pipeline(self):
        """Create the pyannote VAD pipeline on first use."""
        if self._pipeline is None:
            from pyannote.audio import Pipeline

            logger.info("Loading pyannote VAD model: %s", self.config.model_id)
            self._pipeline = Pipeline.from_pretrained(
                self.config.model_id,
                use_auth_token=self._auth_token,
            )
            if self.device.type == "cuda":
                self._pipeline.to(torch.device("cuda"))

            params = {}
            if self.config.min_duration_on is not None:
                params["min_duration_on"] = self.config.min_duration_on
            if self.config.min_duration_off is not None:
                params["min_duration_off"] = self.config.min_duration_off
            if params:
                try:
                    self._pipeline.instantiate(params)
                except Exception as exc:  # pragma: no cover - depends on pyannote model version
                    logger.warning("Could not set pyannote VAD parameters %s: %s", params, exc)

        return self._pipeline

    @property
    def _auth_token(self) -> str | None:
        return (
            self.config.auth_token
            or os.getenv("PYANNOTE_AUTH_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HF_TOKEN")
        )

    def detect_speech(self, audio_path: str | Path) -> list[SpeechRegion]:
        """Return speech regions as start/end dictionaries in seconds."""
        speech = self.pipeline(str(audio_path))
        timeline = speech.get_timeline().support()
        regions = [
            {"start": float(segment.start), "end": float(segment.end), "score": None}
            for segment in timeline
        ]
        return normalize_speech_regions(regions)


def normalize_speech_regions(regions: list[dict[str, Any]]) -> list[SpeechRegion]:
    """Validate and sort speech regions from VAD or fixtures."""
    normalized = []
    for region in regions:
        start = float(region["start"])
        end = float(region["end"])
        if end <= start:
            continue
        score = region.get("score")
        normalized.append(
            {
                "start": start,
                "end": end,
                "score": float(score) if score is not None else None,
            }
        )
    return sorted(normalized, key=lambda item: (float(item["start"]), float(item["end"])))


def merge_speech_regions(
    regions: list[dict[str, Any]],
    max_gap_s: float = 0.5,
) -> list[SpeechRegion]:
    """Merge adjacent speech regions separated by short silence gaps."""
    normalized = normalize_speech_regions(regions)
    if not normalized:
        return []

    merged = [dict(normalized[0])]
    for region in normalized[1:]:
        previous = merged[-1]
        gap = float(region["start"]) - float(previous["end"])
        if gap <= max_gap_s:
            previous["end"] = max(float(previous["end"]), float(region["end"]))
            previous["score"] = _merge_scores(previous.get("score"), region.get("score"))
        else:
            merged.append(dict(region))
    return merged


def split_long_speech_regions(
    regions: list[dict[str, Any]],
    max_duration_s: float | None = None,
) -> list[SpeechRegion]:
    """Split speech regions longer than max_duration_s into contiguous chunks."""
    normalized = normalize_speech_regions(regions)
    if max_duration_s is None:
        return normalized
    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")

    split_regions = []
    for region in normalized:
        start = float(region["start"])
        end = float(region["end"])
        cursor = start
        while cursor < end:
            chunk_end = min(end, cursor + max_duration_s)
            split_regions.append({"start": cursor, "end": chunk_end, "score": region.get("score")})
            cursor = chunk_end
    return split_regions


def prepare_asr_speech_regions(
    regions: list[dict[str, Any]],
    merge_gap_s: float = 0.5,
    max_duration_s: float | None = None,
) -> list[SpeechRegion]:
    """Apply the shared region policy before ASR transcription."""
    merged = merge_speech_regions(regions, max_gap_s=merge_gap_s)
    return split_long_speech_regions(merged, max_duration_s=max_duration_s)


def _merge_scores(left: Any, right: Any) -> float | None:
    scores = [float(score) for score in (left, right) if score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)
