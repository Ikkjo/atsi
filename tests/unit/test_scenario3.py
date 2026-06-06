from pathlib import Path

import numpy as np
import pytest
import torch

from src.diarization.scenario3 import ReferenceIdentifier, load_reference_embeddings


def test_reference_identifier_classify_near_a():
    refs = {"A": torch.tensor([1.0, 0.0]), "B": torch.tensor([0.0, 1.0])}
    identifier = ReferenceIdentifier(refs)
    emb = torch.tensor([[1.0, 0.1]])
    assigned, sims = identifier.classify(emb)
    assert assigned[0] == "A"
    assert sims.shape == (1, 2)


def test_reference_identifier_classify_near_b():
    refs = {"A": torch.tensor([1.0, 0.0]), "B": torch.tensor([0.0, 1.0])}
    identifier = ReferenceIdentifier(refs)
    emb = torch.tensor([[0.1, 1.0]])
    assigned, sims = identifier.classify(emb)
    assert assigned[0] == "B"


def test_reference_identifier_threshold_rejects_uncertain():
    refs = {"A": torch.tensor([1.0, 0.0]), "B": torch.tensor([0.0, 1.0])}
    identifier = ReferenceIdentifier(refs, threshold=0.99)
    # 45-degree vector -> cosine similarity ~0.707
    emb = torch.tensor([[1.0, 1.0]])
    assigned, sims = identifier.classify(emb)
    assert assigned[0] == "unknown"


def test_reference_identifier_classify_segments():
    refs = {"A": torch.tensor([1.0, 0.0])}
    identifier = ReferenceIdentifier(refs)
    segments = [{"start": 0.0, "end": 1.0, "segment_id": "s0"}]
    embeddings = torch.tensor([[1.0, 0.0]])
    result = identifier.classify_segments(embeddings, segments)
    assert result[0]["speaker_id"] == "A"
    assert result[0]["start"] == 0.0
    assert result[0]["segment_id"] == "s0"


def test_reference_identifier_1d_input():
    refs = {"A": torch.tensor([1.0, 0.0])}
    identifier = ReferenceIdentifier(refs)
    emb = torch.tensor([1.0, 0.0])
    assigned, sims = identifier.classify(emb)
    assert len(assigned) == 1
    assert sims.shape == (1, 1)


def test_reference_identifier_np_array_input():
    refs = {"A": np.array([1.0, 0.0]), "B": np.array([0.0, 1.0])}
    identifier = ReferenceIdentifier(refs)
    emb = np.array([[0.0, 1.0]])
    assigned, sims = identifier.classify(emb)
    assert assigned[0] == "B"


def test_load_reference_embeddings(tmp_path):
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    torch.save({"embedding": torch.tensor([1.0, 2.0])}, ref_dir / "ES2002a_A.pt")
    torch.save({"embedding": torch.tensor([3.0, 4.0])}, ref_dir / "ES2002a_B.pt")

    refs = load_reference_embeddings(ref_dir, "ES2002a")
    assert set(refs.keys()) == {"A", "B"}
    assert torch.equal(refs["A"], torch.tensor([1.0, 2.0]))


def test_load_reference_embeddings_missing_meeting(tmp_path):
    refs = load_reference_embeddings(tmp_path, "NONEXISTENT")
    assert refs == {}
