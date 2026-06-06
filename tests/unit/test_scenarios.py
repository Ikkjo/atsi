import numpy as np
import pytest
import torch

from src.diarization.scenarios import (
    run_scenario1_unknown_speakers,
    run_scenario2_oracle_speakers,
    run_scenario3_reference_identification,
)


def test_run_scenario1_unknown_speakers():
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
        {"start": 3.0, "end": 4.0},
    ]
    embeddings = np.array(
        [
            [1, 0],
            [1.1, 0],
            [0, 1],
            [0, 1.1],
        ]
    )
    reference_segments = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 2.0},
        {"speaker_id": "B", "begin_time": 2.0, "end_time": 4.0},
    ]
    result = run_scenario1_unknown_speakers(
        segments, embeddings, reference_segments, distance_threshold=0.5
    )
    assert "diarization_segments" in result
    assert len(result["diarization_segments"]) == 4
    assert len(np.unique(result["cluster_labels"])) == 2


def test_run_scenario1_missing_threshold_raises():
    segments = [{"start": 0.0, "end": 1.0}]
    embeddings = np.array([[1.0]])
    reference_segments = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 1.0}
    ]
    with pytest.raises(ValueError):
        run_scenario1_unknown_speakers(segments, embeddings, reference_segments)


def test_run_scenario2_oracle_speakers():
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 3.0},
        {"start": 3.0, "end": 4.0},
    ]
    embeddings = np.array(
        [
            [1, 0],
            [1.1, 0],
            [0, 1],
            [0, 1.1],
        ]
    )
    reference_segments = [
        {"speaker_id": "A", "begin_time": 0.0, "end_time": 2.0},
        {"speaker_id": "B", "begin_time": 2.0, "end_time": 4.0},
    ]
    result = run_scenario2_oracle_speakers(
        segments, embeddings, reference_segments, n_speakers=2
    )
    assert len(result["diarization_segments"]) == 4
    assert len(np.unique(result["cluster_labels"])) == 2


def test_run_scenario2_invalid_n_speakers():
    with pytest.raises(ValueError):
        run_scenario2_oracle_speakers([], np.array([]), [], n_speakers=0)


def test_run_scenario3_reference_identification():
    segments = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
    ]
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    references = {
        "A": torch.tensor([1.0, 0.0]),
        "B": torch.tensor([0.0, 1.0]),
    }
    result = run_scenario3_reference_identification(segments, embeddings, references)
    assert len(result["diarization_segments"]) == 2
    assert result["diarization_segments"][0]["speaker_id"] == "A"
    assert result["diarization_segments"][1]["speaker_id"] == "B"
    assert result["similarities"].shape == (2, 2)


def test_run_scenario3_with_threshold():
    segments = [
        {"start": 0.0, "end": 1.0},
    ]
    embeddings = torch.tensor([[1.0, 1.0]])
    references = {
        "A": torch.tensor([1.0, 0.0]),
        "B": torch.tensor([0.0, 1.0]),
    }
    result = run_scenario3_reference_identification(
        segments, embeddings, references, threshold=0.99
    )
    assert result["diarization_segments"][0]["speaker_id"] == "unknown"
