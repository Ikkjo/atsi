import numpy as np
import pytest
import torch

from src.diarization.clustering import (
    ahc_clustering,
    cluster_labels_to_speaker_segments,
    compute_overlap_matrix,
    compute_pseudo_der,
    map_clusters_to_speakers,
    sweep_threshold,
)


def test_ahc_clustering_oracle_two_clusters():
    embeddings = np.array(
        [
            [1, 0],
            [1.1, 0],
            [0.9, 0],
            [0, 1],
            [0, 1.1],
            [0, 0.9],
        ]
    )
    labels = ahc_clustering(
        embeddings, n_clusters=2, linkage_method="average", metric="euclidean"
    )
    assert len(labels) == 6
    assert len(np.unique(labels)) == 2


def test_ahc_clustering_auto_threshold_high_merge():
    embeddings = np.array(
        [
            [1, 0],
            [1.1, 0],
            [0, 1],
            [0, 1.1],
        ]
    )
    labels = ahc_clustering(
        embeddings,
        distance_threshold=10.0,
        linkage_method="average",
        metric="euclidean",
    )
    assert len(np.unique(labels)) == 1


def test_ahc_clustering_auto_threshold_low_split():
    embeddings = np.array(
        [
            [1, 0],
            [1.1, 0],
            [0, 1],
            [0, 1.1],
        ]
    )
    # Euclidean distance between [1,0] and [1.1,0] is 0.1.
    # A threshold of 0.15 merges the two close pairs while keeping the two
    # groups separate (average inter-group distance ~1.48).
    labels = ahc_clustering(
        embeddings,
        distance_threshold=0.15,
        linkage_method="average",
        metric="euclidean",
    )
    assert len(np.unique(labels)) == 2


def test_ahc_clustering_empty():
    labels = ahc_clustering(np.array([]), n_clusters=1)
    assert len(labels) == 0


def test_ahc_clustering_single():
    labels = ahc_clustering(np.array([[1.0, 0.0]]), n_clusters=1)
    assert labels[0] == 1


def test_ahc_clustering_mutual_exclusion():
    with pytest.raises(ValueError):
        ahc_clustering(np.array([[1.0]]), n_clusters=1, distance_threshold=0.5)
    with pytest.raises(ValueError):
        ahc_clustering(np.array([[1.0]]))


def test_compute_overlap_matrix():
    cluster_labels = np.array([1, 1, 2])
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
    ]
    reference_segments = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 1.5},
        {"speaker_id": "B", "begin_time": 2.0, "end_time": 3.0},
    ]
    overlap = compute_overlap_matrix(cluster_labels, segments, reference_segments)
    assert overlap.shape == (2, 2)
    # Cluster 1 overlaps A for 1.5 s, B for 0 s
    assert overlap[0, 0] == 1.5
    assert overlap[0, 1] == 0.0
    # Cluster 2 overlaps A for 0 s, B for 1.0 s
    assert overlap[1, 0] == 0.0
    assert overlap[1, 1] == 1.0


def test_map_clusters_to_speakers():
    cluster_labels = np.array([1, 1, 2])
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
    ]
    reference_segments = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 1.5},
        {"speaker_id": "B", "begin_time": 2.0, "end_time": 3.0},
    ]
    mapping = map_clusters_to_speakers(cluster_labels, segments, reference_segments)
    assert mapping[1] == "A"
    assert mapping[2] == "B"


def test_map_clusters_to_speakers_extra_cluster_maps_unknown():
    cluster_labels = np.array([1, 2, 3])
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
    ]
    reference_segments = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 1.0},
        {"speaker_id": "B", "begin_time": 1.0, "end_time": 2.0},
    ]
    mapping = map_clusters_to_speakers(cluster_labels, segments, reference_segments)
    # Cluster 3 has no overlap with either speaker -> unknown
    assert mapping[3] == "unknown"


def test_cluster_labels_to_speaker_segments():
    cluster_labels = np.array([1, 2])
    segments = [
        {"start": 0.0, "end": 1.0, "segment_id": "s0"},
        {"start": 1.0, "end": 2.0, "segment_id": "s1"},
    ]
    mapping = {1: "A", 2: "B"}
    result = cluster_labels_to_speaker_segments(cluster_labels, segments, mapping)
    assert result[0]["speaker_id"] == "A"
    assert result[1]["speaker_id"] == "B"
    assert result[0]["segment_id"] == "s0"


def test_compute_pseudo_der_perfect():
    predicted = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "B"},
    ]
    reference = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 1.0},
        {"speaker_id": "B", "begin_time": 1.0, "end_time": 2.0},
    ]
    assert compute_pseudo_der(predicted, reference) == 0.0


def test_compute_pseudo_der_fully_wrong():
    predicted = [{"start": 0.0, "end": 1.0, "speaker_id": "B"}]
    reference = [{"speaker_id": "A", "begin_time": 0.0, "end_time": 1.0}]
    assert compute_pseudo_der(predicted, reference) == 1.0


def test_compute_pseudo_der_empty():
    assert compute_pseudo_der([], []) == 0.0


def test_sweep_threshold_selects_best():
    # Two synthetic meetings
    embeddings_list = [
        np.array([[1, 0], [1.1, 0], [0, 1], [0, 1.1]]),
        np.array([[1, 0], [1.1, 0]]),
    ]
    segments_list = [
        [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
            {"start": 2.0, "end": 3.0},
            {"start": 3.0, "end": 4.0},
        ],
        [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ],
    ]
    reference_segments_list = [
        [
            {"speaker_id": "A", "begin_time": 0.0, "end_time": 2.0},
            {"speaker_id": "B", "begin_time": 2.0, "end_time": 4.0},
        ],
        [
            {"speaker_id": "A", "begin_time": 0.0, "end_time": 2.0},
        ],
    ]
    thresholds = [0.05, 0.5, 5.0]
    best_threshold, results = sweep_threshold(
        embeddings_list,
        segments_list,
        reference_segments_list,
        thresholds,
        metric="euclidean",
    )
    assert best_threshold is not None
    assert "best_pseudo_der" in results
    assert len(results["sweep"]) == 3
    # With Euclidean distances:
    #   0.05 -> 4 clusters for meeting 1 (high pseudo-DER)
    #   0.5  -> 2 clusters for meeting 1 (perfect)
    #   5.0  -> 1 cluster for meeting 1 (50% pseudo-DER)
    # 0.5 is the clear sweet spot.
    assert best_threshold == 0.5
