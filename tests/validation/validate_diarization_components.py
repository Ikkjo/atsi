"""Smoke/validation script for diarization pipeline components.

Validates that the shared VAD, segmenter, and ECAPA embedding extractor
produce well-formed outputs (no NaNs, reasonable shapes, etc.) on a
real audio file or a synthetic waveform.

Usage:
    python -m tests.validation.validate_diarization_components [audio_path]

If *audio_path* is omitted, a synthetic 5-second multi-speaker signal is
used so the script can run in CI without AMI data.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Ensure src is on path when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.diarization.embeddings import ECAPAEmbeddingConfig, ECAPAEmbeddingExtractor
from src.diarization.segmentation import DiarizationSegmenter, EmbeddingSegmentationConfig
from src.diarization.vad import PyannoteVAD, normalize_speech_regions
from src.utils.hardware import get_device

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_synthetic_audio(path: Path, duration_s: float = 5.0, sr: int = 16000) -> Path:
    """Create a synthetic stereo waveform with two distinct frequency bands."""
    samples = int(duration_s * sr)
    t = torch.linspace(0, duration_s, samples)

    # Left channel: 300 Hz tone, Right channel: 900 Hz tone
    left = torch.sin(2 * math.pi * 300 * t)
    right = torch.sin(2 * math.pi * 900 * t)
    stereo = torch.stack([left, right], dim=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), stereo, sr)
    logger.info("Created synthetic audio: %s", path)
    return path


def validate_vad(audio_path: str | Path) -> list[dict]:
    """Run VAD and assert output sanity."""
    logger.info("--- Validating VAD ---")
    vad = PyannoteVAD()
    regions = vad.detect_speech(audio_path)

    assert isinstance(regions, list), "VAD must return a list"
    assert len(regions) > 0, "VAD must detect at least one speech region"

    for i, region in enumerate(regions):
        assert "start" in region and "end" in region, f"Region {i} missing start/end"
        assert float(region["end"]) > float(region["start"]), f"Region {i} has invalid duration"

    total_speech = sum(float(r["end"]) - float(r["start"]) for r in regions)
    logger.info("VAD: %d regions, %.2f s total speech", len(regions), total_speech)
    return regions


def validate_segmenter(regions: list[dict]) -> list[dict]:
    """Run segmenter and assert output sanity."""
    logger.info("--- Validating Segmenter ---")
    config = EmbeddingSegmentationConfig(clip_duration_s=1.5, min_duration_s=0.5)
    from src.diarization.segmentation import prepare_embedding_segments

    segments = prepare_embedding_segments(regions, config=config)

    assert isinstance(segments, list), "Segmenter must return a list"
    assert len(segments) > 0, "Segmenter must produce at least one segment"

    for i, seg in enumerate(segments):
        assert "start" in seg and "end" in seg, f"Segment {i} missing start/end"
        assert float(seg["end"]) > float(seg["start"]), f"Segment {i} has invalid duration"
        dur = float(seg["end"]) - float(seg["start"])
        assert dur >= config.min_duration_s, f"Segment {i} too short ({dur:.2f}s)"
        assert seg.get("segment_id") is not None, f"Segment {i} missing segment_id"

    logger.info("Segmenter: %d segments", len(segments))
    return segments


def validate_embeddings(audio_path: str | Path, segments: list[dict]) -> dict:
    """Run ECAPA embedding extractor and assert output sanity."""
    logger.info("--- Validating Embedding Extractor ---")
    config = ECAPAEmbeddingConfig()
    extractor = ECAPAEmbeddingExtractor(config=config)

    result = extractor.extract_embeddings(audio_path, segments=segments)

    embeddings = result["embeddings"]
    meta = result["metadata"]
    kept_segments = result["segments"]

    assert isinstance(embeddings, torch.Tensor), "Embeddings must be a torch.Tensor"
    assert embeddings.ndim == 2, f"Embeddings must be 2-D, got {embeddings.ndim}"
    assert embeddings.shape[0] == len(kept_segments), "Mismatch between embeddings and segments"
    assert embeddings.shape[1] > 0, "Embedding dimension must be > 0"

    # No NaNs or Infs
    assert not torch.isnan(embeddings).any(), "Embeddings contain NaN values"
    assert not torch.isinf(embeddings).any(), "Embeddings contain Inf values"

    # Normalised embeddings should have unit norm (approx)
    if config.normalize:
        norms = torch.linalg.vector_norm(embeddings, dim=1)
        mean_norm = float(norms.mean())
        assert 0.9 < mean_norm < 1.1, f"Mean L2 norm {mean_norm:.3f} far from 1.0"

    logger.info(
        "Embeddings: shape=%s, dim=%d, model=%s",
        tuple(embeddings.shape),
        meta["embedding_dim"],
        meta["model_id"],
    )
    return result


def validate_pipeline(audio_path: str | Path) -> bool:
    """Run the full VAD -> segmenter -> embedding pipeline and validate."""
    logger.info("=== Validating Diarization Pipeline ===")
    logger.info("Audio: %s", audio_path)

    try:
        regions = validate_vad(audio_path)
        segments = validate_segmenter(regions)
        result = validate_embeddings(audio_path, segments)

        # Summary
        logger.info(
            "Pipeline OK: %d speech regions -> %d segments -> %d embeddings (%d-D)",
            len(regions),
            len(segments),
            result["embeddings"].shape[0],
            result["embeddings"].shape[1],
        )
        return True
    except Exception as exc:
        logger.error("Pipeline validation failed: %s", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate diarization pipeline components")
    parser.add_argument("audio_path", nargs="?", type=str, help="Path to a 16kHz WAV file")
    args = parser.parse_args()

    if args.audio_path:
        audio_path = Path(args.audio_path)
        if not audio_path.exists():
            logger.error("Audio file not found: %s", audio_path)
            return 1
    else:
        audio_path = Path("/tmp/validate_diarization_synthetic.wav")
        _make_synthetic_audio(audio_path)

    success = validate_pipeline(audio_path)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
