"""Scenario 3: Reference speaker identification via cosine similarity.

Loads per-meeting reference embeddings (produced from an enrollment period) and
classifies each diarization segment by nearest cosine similarity.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.diarization.embeddings import l2_normalize

logger = logging.getLogger(__name__)


class ReferenceIdentifier:
    """Classify segment embeddings against a set of reference embeddings.

    Embeddings are expected to be L2-normalised (or will be normalised
    internally) so that cosine similarity reduces to a dot product.
    """

    def __init__(
        self,
        references: dict[str, torch.Tensor | np.ndarray],
        threshold: float | None = None,
    ) -> None:
        """Args:
            references: Mapping from speaker ID to reference embedding vector.
            threshold: Optional minimum cosine similarity. If the best score is
                below this value the segment is labelled ``"unknown"``.
        """
        self.references = {
            spk: torch.as_tensor(emb, dtype=torch.float32)
            for spk, emb in references.items()
        }
        self.threshold = threshold
        self._ref_matrix: torch.Tensor | None = None

    @property
    def speaker_ids(self) -> list[str]:
        """Ordered list of speaker IDs corresponding to reference rows."""
        return sorted(self.references)

    @property
    def ref_matrix(self) -> torch.Tensor:
        """Stacked, L2-normalised reference matrix of shape ``(S, D)``."""
        if self._ref_matrix is None:
            self._ref_matrix = torch.stack(
                [l2_normalize(self.references[spk]) for spk in self.speaker_ids]
            )
        return self._ref_matrix

    def classify(
        self,
        embeddings: torch.Tensor | np.ndarray,
    ) -> tuple[list[str], torch.Tensor]:
        """Classify each segment embedding to the nearest reference speaker.

        Args:
            embeddings: Tensor of shape ``(N, D)`` or ``(D,)``.

        Returns:
            Tuple of ``(assigned_speaker_ids, similarity_matrix)`` where
            *similarity_matrix* has shape ``(N, S)`` and *assigned_speaker_ids*
            is a list of length ``N``.
        """
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)

        # L2-normalise so cosine similarity = dot product
        normalised = torch.stack([l2_normalize(emb) for emb in embeddings])
        similarities = torch.matmul(normalised, self.ref_matrix.T)

        best_scores, best_indices = similarities.max(dim=1)
        speaker_ids = self.speaker_ids

        assigned: list[str] = []
        for score, idx in zip(best_scores, best_indices):
            spk = speaker_ids[int(idx)]
            if self.threshold is not None and float(score) < self.threshold:
                assigned.append("unknown")
            else:
                assigned.append(spk)

        return assigned, similarities

    def classify_segments(
        self,
        segment_embeddings: torch.Tensor | np.ndarray,
        segments: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Classify a list of segments and return copies with ``speaker_id`` added.

        Args:
            segment_embeddings: Tensor of shape ``(N, D)``.
            segments: List of segment dicts (length ``N``).

        Returns:
            List of segment dicts with added ``speaker_id`` key.
        """
        assigned, _ = self.classify(segment_embeddings)
        result: list[dict[str, Any]] = []
        for spk, seg in zip(assigned, segments):
            result.append(
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "speaker_id": spk,
                    "segment_id": seg.get("segment_id"),
                }
            )
        return result


def load_reference_embeddings(
    reference_dir: str | Path,
    meeting_id: str,
) -> dict[str, torch.Tensor]:
    """Load reference embeddings saved by ``ReferenceEmbeddingExtractor``.

    Expects files named ``{meeting_id}_{speaker_id}.pt`` where each file contains
    a dict with an ``"embedding"`` key.

    Args:
        reference_dir: Directory containing ``.pt`` reference files.
        meeting_id: Meeting identifier used as the filename prefix.

    Returns:
        Mapping from speaker ID to embedding tensor.
    """
    reference_dir = Path(reference_dir)
    pattern = f"{meeting_id}_*.pt"
    references: dict[str, torch.Tensor] = {}

    for path in reference_dir.glob(pattern):
        # Filename: {meeting_id}_{speaker_id}.pt
        speaker_id = path.stem[len(meeting_id) + 1 :]
        data = torch.load(path, map_location="cpu", weights_only=False)
        references[speaker_id] = data["embedding"]

    return references
