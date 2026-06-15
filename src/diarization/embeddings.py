"""Speaker embedding extraction with SpeechBrain ECAPA-TDNN."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.data.preprocessing import TARGET_SR, load_audio
from src.diarization.segmentation import (
    DiarizationSegmenter,
    EmbeddingSegment,
    EmbeddingSegmentationConfig,
)
from src.utils.hardware import get_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ECAPAEmbeddingConfig:
    """Configuration for SpeechBrain ECAPA embedding extraction."""

    model_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    savedir: str = "pretrained_models/spkrec-ecapa-voxceleb"
    target_sr: int = TARGET_SR
    normalize: bool = True
    min_samples: int = int(0.2 * TARGET_SR)


class ECAPAEmbeddingExtractor:
    """Lazy-loading ECAPA-TDNN extractor for diarization segments."""

    def __init__(
        self,
        config: ECAPAEmbeddingConfig | None = None,
        segmenter: DiarizationSegmenter | None = None,
    ) -> None:
        self.config = config or ECAPAEmbeddingConfig()
        self.device = get_device()
        self.segmenter = segmenter or DiarizationSegmenter(
            config=EmbeddingSegmentationConfig()
        )
        self._model = None

    @property
    def model(self):
        """Load SpeechBrain's ECAPA-TDNN model on first use."""
        if self._model is None:
            from speechbrain.inference.classifiers import EncoderClassifier

            logger.info("Loading ECAPA-TDNN speaker embedding model: %s", self.config.model_id)
            device = "cuda:0" if self.device.type == "cuda" else str(self.device)
            self._model = EncoderClassifier.from_hparams(
                source=self.config.model_id,
                savedir=self.config.savedir,
                run_opts={"device": device},
            )
        return self._model

    def extract_embeddings(
        self,
        audio_path: str | Path,
        segments: list[dict[str, Any]] | None = None,
        meeting_id: str | None = None,
        output_dir: str | Path | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """Extract and optionally cache ECAPA embeddings for one recording."""
        audio_path = Path(audio_path)
        cache_path = None
        if output_dir is not None:
            cache_path = self.result_path(audio_path, output_dir, meeting_id=meeting_id)
            if use_cache and cache_path.exists():
                logger.info("Using cached speaker embeddings: %s", cache_path)
                return self.load_result(cache_path)

        prepared_segments = (
            [dict(segment) for segment in segments]
            if segments is not None
            else self.segmenter.detect_segments(audio_path)
        )

        embeddings: list[torch.Tensor] = []
        kept_segments: list[EmbeddingSegment] = []
        for index, segment in enumerate(prepared_segments):
            start = float(segment["start"])
            end = float(segment["end"])
            waveform, _ = load_audio(
                audio_path,
                target_sr=self.config.target_sr,
                start_time=start,
                end_time=end,
            )
            if waveform.shape[-1] < self.config.min_samples:
                logger.debug("Skipping too-short embedding segment %.2f-%.2f", start, end)
                continue

            embedding = self.compute_embedding(waveform)
            embeddings.append(embedding)

            kept = dict(segment)
            kept.setdefault("segment_id", f"seg_{index:06d}")
            kept["start"] = start
            kept["end"] = end
            kept["duration"] = end - start
            kept["embedding_index"] = len(embeddings) - 1
            kept_segments.append(kept)

        embedding_tensor = torch.stack(embeddings).cpu() if embeddings else torch.empty((0, 0))
        result = {
            "metadata": {
                "audio_path": str(audio_path),
                "meeting_id": meeting_id,
                "model_id": self.config.model_id,
                "sample_rate": self.config.target_sr,
                "embedding_dim": int(embedding_tensor.shape[1]) if embedding_tensor.ndim == 2 else 0,
                "normalized": self.config.normalize,
            },
            "segments": kept_segments,
            "embeddings": embedding_tensor,
        }

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, cache_path)
            logger.info("Saved speaker embeddings to %s", cache_path)

        return result

    def compute_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute one ECAPA embedding and record its actual output dimension."""
        if waveform.ndim != 2:
            raise ValueError("waveform must have shape (channels, samples)")
        if waveform.shape[0] != 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(waveform).squeeze()
        embedding = embedding.detach().float().cpu().reshape(-1)
        return l2_normalize(embedding) if self.config.normalize else embedding

    @staticmethod
    def result_path(
        audio_path: str | Path,
        output_dir: str | Path,
        meeting_id: str | None = None,
    ) -> Path:
        """Return the cache path for segment embeddings."""
        stem = meeting_id or Path(audio_path).stem
        return Path(output_dir) / f"{stem}_ecapa_embeddings.pt"

    @staticmethod
    def load_result(path: str | Path) -> dict[str, Any]:
        """Load a cached embedding result."""
        return torch.load(path, map_location="cpu", weights_only=False)


def l2_normalize(embedding: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return an L2-normalized copy of an embedding tensor."""
    norm = torch.linalg.vector_norm(embedding.float())
    if norm <= eps:
        return torch.zeros_like(embedding.float())
    return embedding.float() / norm
