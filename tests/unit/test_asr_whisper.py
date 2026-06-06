import json
from pathlib import Path

from src.asr.whisper import (
    WhisperASR,
    WhisperASRConfig,
    _extract_segments,
    _extract_words,
    _fallback_word_timestamps,
)


def test_extract_segments_normalizes_huggingface_chunks():
    raw = {
        "chunks": [
            {"timestamp": (0.0, 1.5), "text": " hello "},
            {"timestamp": [1.5, 3.0], "text": "world"},
        ]
    }

    assert _extract_segments(raw) == [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "hello"},
        {"id": 1, "start": 1.5, "end": 3.0, "text": "world"},
    ]


def test_extract_words_skips_empty_chunks():
    raw = {
        "chunks": [
            {"timestamp": (0.0, 0.4), "text": " hello "},
            {"timestamp": (0.4, 0.5), "text": " "},
        ]
    }

    assert _extract_words(raw) == [
        {"id": 0, "start": 0.0, "end": 0.4, "word": "hello", "source": "native"}
    ]


def test_fallback_word_timestamps_stay_inside_segment_boundaries():
    words = _fallback_word_timestamps(
        [
            {"id": 0, "start": 10.0, "end": 12.0, "text": "a longer"},
            {"id": 1, "start": 12.0, "end": 13.0, "text": "word"},
        ]
    )

    assert [word["word"] for word in words] == ["a", "longer", "word"]
    assert words[0]["start"] == 10.0
    assert words[-1]["end"] == 13.0
    assert all(words[idx]["end"] <= words[idx + 1]["start"] for idx in range(len(words) - 1))


def test_result_path_uses_meeting_id():
    path = WhisperASR.result_path("audio.wav", "results/asr/raw", meeting_id="EN2001a")

    assert path == Path("results/asr/raw/EN2001a_whisper.json")


def test_transcribe_uses_cache_without_loading_audio(tmp_path):
    cached = {"meeting_id": "M1", "text": "cached"}
    cache_path = tmp_path / "M1_whisper.json"
    cache_path.write_text(json.dumps(cached))

    asr = WhisperASR(WhisperASRConfig(model_id="test-model"))
    result = asr.transcribe(
        tmp_path / "missing.wav",
        meeting_id="M1",
        output_dir=tmp_path,
        use_cache=True,
    )

    assert result == cached


def test_transcribe_speech_regions_offsets_region_local_timestamps(monkeypatch, tmp_path):
    monkeypatch.setattr("src.asr.whisper.get_audio_duration", lambda _: 99.0)

    class FakeASR(WhisperASR):
        def __init__(self):
            super().__init__(WhisperASRConfig(model_id="test-model", batch_size=1))
            self.calls = []

        def transcribe(self, audio_path, meeting_id=None, output_dir=None, start_time=None, end_time=None, use_cache=False):
            self.calls.append((start_time, end_time))
            return {
                "start_time": start_time,
                "end_time": end_time,
                "text": f"region-{len(self.calls)}",
                "segments": [{"id": 0, "start": 0.25, "end": 1.0, "text": "hello"}],
                "words": [{"id": 0, "start": 0.25, "end": 0.6, "word": "hello", "source": "native"}],
                "word_timestamp_mode": "native",
            }

    asr = FakeASR()
    result = asr.transcribe_speech_regions(
        tmp_path / "missing.wav",
        [
            {"start": 10.0, "end": 11.0},
            {"start": 11.2, "end": 12.0},
            {"start": 20.0, "end": 21.0},
        ],
        meeting_id="M1",
        merge_gap_s=0.5,
    )

    assert asr.calls == [(10.0, 12.0), (20.0, 21.0)]
    assert result["segments"][0]["start"] == 10.25
    assert result["segments"][1]["end"] == 21.0
    assert result["words"][0]["start"] == 10.25
    assert result["vad"]["transcribed_regions"] == [
        {"start": 10.0, "end": 12.0, "score": None},
        {"start": 20.0, "end": 21.0, "score": None},
    ]
    assert result["text"] == "region-1 region-2"
