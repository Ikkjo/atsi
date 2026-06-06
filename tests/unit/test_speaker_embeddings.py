from pathlib import Path

import torch

from src.diarization.embeddings import ECAPAEmbeddingConfig, ECAPAEmbeddingExtractor, l2_normalize


class FakeECAPAModel:
    def encode_batch(self, waveform):
        total = waveform.sum().reshape(1, 1, 1)
        return torch.cat([total, total + 3.0], dim=-1)


class FakeExtractor(ECAPAEmbeddingExtractor):
    @property
    def model(self):
        return FakeECAPAModel()


def test_l2_normalize_matches_hand_computed_vector():
    normalized = l2_normalize(torch.tensor([3.0, 4.0]))

    assert torch.allclose(normalized, torch.tensor([0.6, 0.8]))


def test_l2_normalize_returns_zero_for_zero_vector():
    assert torch.equal(l2_normalize(torch.zeros(3)), torch.zeros(3))


def test_compute_embedding_converts_to_mono_and_normalizes():
    extractor = FakeExtractor(config=ECAPAEmbeddingConfig(normalize=True))
    waveform = torch.ones((2, 1600))

    embedding = extractor.compute_embedding(waveform)

    assert embedding.shape == (2,)
    assert torch.isclose(torch.linalg.vector_norm(embedding), torch.tensor(1.0))


def test_extract_embeddings_records_actual_dimension_and_saves_cache(tmp_path, monkeypatch):
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"placeholder")

    def fake_load_audio(path, target_sr, start_time, end_time):
        assert Path(path) == audio_path
        assert target_sr == 16_000
        samples = int((end_time - start_time) * target_sr)
        return torch.ones((1, samples)), target_sr

    monkeypatch.setattr("src.diarization.embeddings.load_audio", fake_load_audio)
    extractor = FakeExtractor(config=ECAPAEmbeddingConfig(normalize=False, min_samples=1))

    result = extractor.extract_embeddings(
        audio_path,
        segments=[{"segment_id": "a", "start": 0.0, "end": 1.0}],
        meeting_id="ES2002a",
        output_dir=tmp_path,
    )
    cached = extractor.extract_embeddings(
        audio_path,
        meeting_id="ES2002a",
        output_dir=tmp_path,
        use_cache=True,
    )

    assert result["metadata"]["embedding_dim"] == 2
    assert result["segments"][0]["embedding_index"] == 0
    assert result["embeddings"].shape == (1, 2)
    assert cached["metadata"]["embedding_dim"] == 2
