from src.integration.alignment import align_words_to_speakers, assign_speaker_to_word


def test_assign_speaker_uses_maximum_temporal_overlap():
    word = {"start": 1.4, "end": 2.4, "word": "hello"}
    segments = [
        {"start": 0.0, "end": 1.6, "speaker_id": "Speaker_A"},
        {"start": 1.6, "end": 3.0, "speaker_id": "Speaker_B"},
    ]

    assert assign_speaker_to_word(word, segments) == "Speaker_B"


def test_assign_speaker_uses_nearest_start_when_word_in_gap():
    word = {"start": 2.1, "end": 2.3, "word": "gap"}
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "Speaker_A"},
        {"start": 3.0, "end": 4.0, "speaker_id": "Speaker_B"},
    ]

    assert assign_speaker_to_word(word, segments) == "Speaker_B"


def test_assign_speaker_uses_nearest_start_for_equal_overlap_tie():
    word = {"start": 1.0, "end": 3.0, "word": "tie"}
    segments = [
        {"start": 0.0, "end": 2.0, "speaker_id": "Speaker_A"},
        {"start": 2.0, "end": 4.0, "speaker_id": "Speaker_B"},
    ]

    assert assign_speaker_to_word(word, segments) == "Speaker_A"


def test_align_words_to_speakers_preserves_word_fields_and_ids():
    words = [
        {"start": 0.0, "end": 0.4, "word": "hello", "source": "native"},
        {"id": 7, "start": 1.2, "end": 1.5, "word": "world"},
    ]
    segments = [
        {"start": 0.0, "end": 1.0, "speaker_id": "Speaker_A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "Speaker_B"},
    ]

    aligned = align_words_to_speakers(words, segments)

    assert aligned[0] == {
        "id": 0,
        "start": 0.0,
        "end": 0.4,
        "word": "hello",
        "source": "native",
        "speaker_id": "Speaker_A",
    }
    assert aligned[1]["id"] == 7
    assert aligned[1]["speaker_id"] == "Speaker_B"


def test_assign_speaker_returns_unknown_without_diarization_segments():
    assert assign_speaker_to_word({"start": 0.0, "end": 1.0}, []) == "unknown"
