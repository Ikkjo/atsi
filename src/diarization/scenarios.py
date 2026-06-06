"""High-level scenario runners for Epic 4 speaker diarization.

Wraps the core clustering and reference-identification logic into convenience
functions that accept pre-computed embeddings and produce labelled diarization
segments ready for RTTM export or integration with ASR.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.diarization.clustering import (
    ahc_clustering,
    cluster_labels_to_speaker_segments,
    map_clusters_to_speakers,
    sweep_threshold,
)
from src.diarization.scenario3 import ReferenceIdentifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenario 1: Unknown number of speakers (AHC auto-k)
# ---------------------------------------------------------------------------


def run_scenario1_unknown_speakers(
    segments: list[dict[str, Any]],
    embeddings: np.ndarray | torch.Tensor,
    reference_segments: list[dict[str, Any]],
    distance_threshold: float | None = None,
    linkage_method: str = "average",
    metric: str = "cosine",
) -> dict[str, Any]:
    """Run Scenario 1: AHC with unknown number of speakers.

    If *distance_threshold* is ``None``, *reference_segments* must be provided
    so that a threshold sweep can be performed. However, this function does **not**
    perform the sweep itself — the caller is responsible for running
    :func:`sweep_threshold` on a validation set and passing the chosen
    *distance_threshold* here.

    Args:
        segments: List of segment dicts with ``start`` and ``end``.
        embeddings: Segment embeddings of shape ``(N, D)``.
        reference_segments: Ground-truth reference segments with ``speaker_id``,
            ``begin_time`` and ``end_time``. Used for optimal cluster-to-speaker
            mapping.
        distance_threshold: Distance threshold for AHC. If ``None``, a
            ``ValueError`` is raised.
        linkage_method: AHC linkage method.
        metric: Distance metric for ``pdist``.

    Returns:
        Dict with keys:
        - ``"diarization_segments"``: List of segments with ``speaker_id``.
        - ``"cluster_labels"``: Array of cluster labels.
        - ``"cluster_to_speaker"``: Mapping dict.
    """
    if distance_threshold is None:
        raise ValueError(
            "distance_threshold is required for Scenario 1. "
            "Run sweep_threshold() on a validation set to select one."
        )

    labels = ahc_clustering(
        embeddings,
        distance_threshold=distance_threshold,
        linkage_method=linkage_method,
        metric=metric,
    )
    mapping = map_clusters_to_speakers(labels, segments, reference_segments)
    diarization = cluster_labels_to_speaker_segments(labels, segments, mapping)

    return {
        "diarization_segments": diarization,
        "cluster_labels": labels,
        "cluster_to_speaker": mapping,
    }


# ---------------------------------------------------------------------------
# Scenario 2: Known number of speakers (oracle AHC)
# ---------------------------------------------------------------------------


def run_scenario2_oracle_speakers(
    segments: list[dict[str, Any]],
    embeddings: np.ndarray | torch.Tensor,
    reference_segments: list[dict[str, Any]],
    n_speakers: int,
    linkage_method: str = "average",
    metric: str = "cosine",
) -> dict[str, Any]:
    """Run Scenario 2: AHC with a fixed number of clusters (oracle).

    The speaker count should be taken from the annotated ground truth for the
    meeting rather than hard-coded.

    Args:
        segments: List of segment dicts with ``start`` and ``end``.
        embeddings: Segment embeddings of shape ``(N, D)``.
        reference_segments: Reference segments for cluster-to-speaker mapping.
        n_speakers: Number of clusters to force (oracle speaker count).
        linkage_method: AHC linkage method.
        metric: Distance metric for ``pdist``.

    Returns:
        Dict with keys:
        - ``"diarization_segments"``: List of segments with ``speaker_id``.
        - ``"cluster_labels"``: Array of cluster labels.
        - ``"cluster_to_speaker"``: Mapping dict.
    """
    if n_speakers <= 0:
        raise ValueError("n_speakers must be positive")

    labels = ahc_clustering(
        embeddings,
        n_clusters=n_speakers,
        linkage_method=linkage_method,
        metric=metric,
    )
    mapping = map_clusters_to_speakers(labels, segments, reference_segments)
    diarization = cluster_labels_to_speaker_segments(labels, segments, mapping)

    return {
        "diarization_segments": diarization,
        "cluster_labels": labels,
        "cluster_to_speaker": mapping,
    }


# ---------------------------------------------------------------------------
# Scenario 3: Reference identification (direct classification)
# ---------------------------------------------------------------------------


def run_scenario3_reference_identification(
    segments: list[dict[str, Any]],
    embeddings: np.ndarray | torch.Tensor,
    references: dict[str, torch.Tensor | np.ndarray],
    threshold: float | None = None,
) -> dict[str, Any]:
    """Run Scenario 3: classify each segment by cosine similarity to references.

    Args:
        segments: List of segment dicts with ``start`` and ``end``.
        embeddings: Segment embeddings of shape ``(N, D)``.
        references: Mapping from speaker ID to reference embedding vector.
        threshold: Optional minimum cosine similarity for a valid assignment.
            Segments below the threshold are labelled ``"unknown"``.

    Returns:
        Dict with keys:
        - ``"diarization_segments"``: List of segments with ``speaker_id``.
        - ``"similarities"``: Similarity matrix of shape ``(N, S)``.
    """
    identifier = ReferenceIdentifier(references=references, threshold=threshold)
    diarization = identifier.classify_segments(embeddings, segments)

    # Also return the full similarity matrix for diagnostics
    _, similarities = identifier.classify(embeddings)

    return {
        "diarization_segments": diarization,
        "similarities": similarities,
    }
