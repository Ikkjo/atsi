from src.evaluation.speaker_identification import evaluate_speaker_identification


def test_speaker_identification_accuracy_and_confusion_matrix():
    reference = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "B"},
    ]
    hypothesis = [
        {"start": 0.0, "end": 1.0, "speaker_id": "A"},
        {"start": 1.0, "end": 2.0, "speaker_id": "A"},
        {"start": 3.0, "end": 4.0, "speaker_id": "B"},
    ]

    result = evaluate_speaker_identification(reference, hypothesis)

    assert result["segment_accuracy"] == 0.5
    assert result["duration_weighted_accuracy"] == 0.5
    assert result["correct_segments"] == 1
    assert result["evaluated_segments"] == 2
    assert result["no_overlap_segments"] == 1
    assert result["confusion_matrix"]["A"]["A"] == 1
    assert result["confusion_matrix"]["B"]["A"] == 1
    assert result["confusion_matrix"]["no_overlap"]["B"] == 1
