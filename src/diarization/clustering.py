"""Agglomerative Hierarchical Clustering for speaker diarization.

Provides AHC with automatic or oracle cluster counts, cluster-to-speaker
mapping via optimal assignment, and a threshold-sweep helper for validation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core clustering
# ---------------------------------------------------------------------------


def ahc_clustering(
    embeddings: np.ndarray | torch.Tensor,
    n_clusters: int | None = None,
    distance_threshold: float | None = None,
    linkage_method: str = "average",
    metric: str = "cosine",
) -> np.ndarray:
    """Run Agglomerative Hierarchical Clustering on speaker embeddings.

    Args:
        embeddings: Array of shape ``(N, D)`` where ``N`` is the number of
            segments and ``D`` is the embedding dimension.
        n_clusters: Fixed number of clusters (oracle mode). Mutually exclusive
            with *distance_threshold*.
        distance_threshold: Distance cut threshold for auto-k (Scenario 1).
            Mutually exclusive with *n_clusters*.
        linkage_method: Linkage method forwarded to ``scipy`` (default:
            ``average``).
        metric: Distance metric forwarded to ``scipy.spatial.distance.pdist``.
            Cosine distance is appropriate for L2-normalised embeddings.

    Returns:
        Cluster labels (1-indexed) of shape ``(N,)``.

    Raises:
        ValueError: If both or neither of *n_clusters* and *distance_threshold*
            are provided.
    """
    if embeddings is None or len(embeddings) == 0:
        return np.array([], dtype=int)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    if (n_clusters is not None) and (distance_threshold is not None):
        raise ValueError("Provide exactly one of n_clusters or distance_threshold")
    if (n_clusters is None) and (distance_threshold is None):
        raise ValueError("Provide exactly one of n_clusters or distance_threshold")

    if embeddings.shape[0] == 1:
        return np.array([1], dtype=int)

    distances = pdist(embeddings, metric=metric)
    Z = linkage(distances, method=linkage_method)

    if n_clusters is not None:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    else:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")

    return labels.astype(int)


# ---------------------------------------------------------------------------
# Cluster-to-speaker mapping
# ---------------------------------------------------------------------------


def _segment_overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """Compute temporal overlap between two segments."""
    return max(0.0, min(end1, end2) - max(start1, start2))


def compute_overlap_matrix(
    cluster_labels: np.ndarray,
    segments: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]],
) -> np.ndarray:
    """Compute total overlap duration between each cluster and reference speaker.

    Args:
        cluster_labels: Array of cluster labels (1-indexed) of shape ``(N,)``.
        segments: List of segment dicts with ``start`` and ``end`` keys.
        reference_segments: List of reference segment dicts with ``speaker_id``,
            ``begin_time`` and ``end_time`` keys.

    Returns:
        Overlap matrix of shape ``(n_clusters, n_speakers)``.
    """
    unique_clusters = np.unique(cluster_labels)
    unique_speakers = sorted({seg["speaker_id"] for seg in reference_segments})

    overlap = np.zeros((len(unique_clusters), len(unique_speakers)), dtype=float)

    for i, cluster in enumerate(unique_clusters):
        cluster_segs = [
            segments[j] for j in range(len(segments)) if cluster_labels[j] == cluster
        ]
        for j, speaker in enumerate(unique_speakers):
            total = 0.0
            for cseg in cluster_segs:
                for rseg in reference_segments:
                    if rseg["speaker_id"] != speaker:
                        continue
                    total += _segment_overlap(
                        float(cseg["start"]),
                        float(cseg["end"]),
                        float(rseg["begin_time"]),
                        float(rseg["end_time"]),
                    )
            overlap[i, j] = total

    return overlap


def map_clusters_to_speakers(
    cluster_labels: np.ndarray,
    segments: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]],
) -> dict[int, str]:
    """Map cluster IDs to reference speaker IDs via optimal assignment.

    Uses ``scipy.optimize.linear_sum_assignment`` on an overlap matrix padded
    with a dummy "unknown" column so that excess clusters can be mapped to
    *unknown* rather than forced onto a real speaker.

    Args:
        cluster_labels: Array of cluster labels (1-indexed).
        segments: List of segment dicts with ``start`` and ``end``.
        reference_segments: List of reference segment dicts with ``speaker_id``.

    Returns:
        Mapping from cluster ID (int) to speaker ID (str). Unmatched clusters
        map to the string ``"unknown"``.
    """
    if len(cluster_labels) == 0:
        return {}

    unique_clusters = np.unique(cluster_labels)
    unique_speakers = sorted({seg["speaker_id"] for seg in reference_segments})

    overlap = compute_overlap_matrix(cluster_labels, segments, reference_segments)

    # Pad with a dummy "unknown" column (zero overlap)
    padded = np.zeros((len(unique_clusters), len(unique_speakers) + 1), dtype=float)
    padded[:, : len(unique_speakers)] = overlap

    # Turn maximisation into minimisation
    max_overlap = padded.max()
    cost = max_overlap - padded if max_overlap > 0 else np.zeros_like(padded)

    row_ind, col_ind = linear_sum_assignment(cost)

    cluster_to_speaker: dict[int, str] = {}
    for r, c in zip(row_ind, col_ind):
        cluster_id = int(unique_clusters[r])
        if c < len(unique_speakers):
            cluster_to_speaker[cluster_id] = unique_speakers[c]
        else:
            cluster_to_speaker[cluster_id] = "unknown"

    return cluster_to_speaker


# ---------------------------------------------------------------------------
# Segment formatting
# ---------------------------------------------------------------------------


def cluster_labels_to_speaker_segments(
    cluster_labels: np.ndarray,
    segments: list[dict[str, Any]],
    cluster_to_speaker: dict[int, str],
) -> list[dict[str, Any]]:
    """Produce diarization segments with speaker IDs assigned from clusters.

    Args:
        cluster_labels: Cluster label per segment.
        segments: Original segment dicts.
        cluster_to_speaker: Mapping from cluster ID to speaker ID.

    Returns:
        List of segment dicts with an added ``speaker_id`` key.
    """
    result: list[dict[str, Any]] = []
    for label, segment in zip(cluster_labels, segments):
        speaker = cluster_to_speaker.get(int(label), "unknown")
        result.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "speaker_id": speaker,
                "segment_id": segment.get("segment_id"),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Lightweight pseudo-DER for threshold sweeps
# ---------------------------------------------------------------------------


def compute_pseudo_der(
    predicted_segments: list[dict[str, Any]],
    reference_segments: list[dict[str, Any]],
) -> float:
    """Compute a lightweight pseudo-DER for threshold selection.

    This approximation considers only **confusion** (mis-assigned speech)
    relative to the total speech duration in the predicted segments.
    Missed speech and false alarm are ignored because the predicted segments
    are derived from the same VAD pipeline; the sweep is focused on the
    clustering quality.

    Args:
        predicted_segments: Predicted segments with ``speaker_id``.
        reference_segments: Reference segments with ``speaker_id``,
            ``begin_time`` and ``end_time``.

    Returns:
        Pseudo-DER in the range ``[0, 1]``.
    """
    total_speech = sum(float(seg["end"]) - float(seg["start"]) for seg in predicted_segments)
    if total_speech == 0:
        return 0.0

    correct = 0.0
    for pseg in predicted_segments:
        assigned_speaker = pseg["speaker_id"]
        for rseg in reference_segments:
            if rseg["speaker_id"] == assigned_speaker:
                correct += _segment_overlap(
                    float(pseg["start"]),
                    float(pseg["end"]),
                    float(rseg["begin_time"]),
                    float(rseg["end_time"]),
                )

    return max(0.0, (total_speech - correct) / total_speech)


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------


def sweep_threshold(
    embeddings_list: list[np.ndarray],
    segments_list: list[list[dict[str, Any]]],
    reference_segments_list: list[list[dict[str, Any]]],
    thresholds: list[float],
    linkage_method: str = "average",
    metric: str = "cosine",
) -> tuple[float, dict[str, Any]]:
    """Sweep distance thresholds on a validation set and pick the best one.

    Args:
        embeddings_list: One embedding array per validation meeting.
        segments_list: One segment list per validation meeting.
        reference_segments_list: One reference segment list per validation meeting.
        thresholds: Candidate distance thresholds to evaluate.
        linkage_method: Linkage method for AHC.
        metric: Distance metric for AHC.

    Returns:
        Tuple of ``(best_threshold, sweep_results)`` where *sweep_results* is a
        dict containing the best threshold, the best pseudo-DER, and per-threshold
        details.
    """
    if len({len(embeddings_list), len(segments_list), len(reference_segments_list)}) != 1:
        raise ValueError("All input lists must have the same length")

    best_threshold = None
    best_der = float("inf")
    sweep_details: list[dict[str, Any]] = []

    for threshold in thresholds:
        ders = []
        for emb, segs, ref in zip(embeddings_list, segments_list, reference_segments_list):
            if len(emb) == 0:
                continue
            labels = ahc_clustering(
                emb,
                distance_threshold=threshold,
                linkage_method=linkage_method,
                metric=metric,
            )
            mapping = map_clusters_to_speakers(labels, segs, ref)
            pred = cluster_labels_to_speaker_segments(labels, segs, mapping)
            der = compute_pseudo_der(pred, ref)
            ders.append(der)

        avg_der = float(np.mean(ders)) if ders else float("inf")
        sweep_details.append(
            {
                "threshold": threshold,
                "avg_pseudo_der": avg_der,
                "per_meeting_der": ders,
            }
        )

        if avg_der < best_der:
            best_der = avg_der
            best_threshold = threshold

    logger.info("Best threshold: %.4f (pseudo-DER=%.4f)", best_threshold, best_der)
    results: dict[str, Any] = {
        "best_threshold": best_threshold,
        "best_pseudo_der": best_der,
        "sweep": sweep_details,
    }
    return best_threshold, results
