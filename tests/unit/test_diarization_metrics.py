from src.evaluation.diarization_metrics import compute_der, compute_jer, evaluate_diarization_metrics, load_diarization_segments


def test_load_diarization_segments_normalizes_annotation_speaker_key():
    segments = load_diarization_segments({"segments": [{"speaker": "A", "start": 0.0, "end": 1.0}]})

    assert segments == [{"speaker": "A", "start": 0.0, "end": 1.0, "duration": 1.0, "speaker_id": "A"}]


def test_compute_der_reports_components_for_speaker_confusion():
    reference = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "B"},
    ]
    hypothesis = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "A"},
    ]

    result = compute_der(reference, hypothesis, collar=0.0, skip_overlap=True)

    assert result["der"] == 0.5
    assert result["components"]["speaker_confusion"] == 1.0
    assert result["components"]["missed_speech"] == 0.0
    assert result["components"]["false_alarm"] == 0.0


def test_compute_jer_and_combined_metrics():
    reference = [{"start": 0.0, "end": 1.0, "speaker_id": "A"}]
    hypothesis = [{"start": 0.0, "end": 1.0, "speaker_id": "A"}]

    assert compute_jer(reference, hypothesis)["jer"] == 0.0
    assert evaluate_diarization_metrics(reference, hypothesis)["der"]["der"] == 0.0
