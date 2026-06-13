"""Scenario 3 speaker-identification accuracy utilities."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.evaluation.diarization_metrics import load_diarization_segments, save_metrics_json


def evaluate_speaker_identification(
    reference: str | Path | dict[str, Any] | list[dict[str, Any]],
    hypothesis: str | Path | dict[str, Any] | list[dict[str, Any]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate predicted Scenario 3 segment identities against reference segments.

    Each predicted segment is matched to the reference speaker with maximum time
    overlap. The primary accuracy is segment-weighted; a duration-weighted score
    is also returned for analysis.
    """
    reference_segments = load_diarization_segments(reference)
    hypothesis_segments = load_diarization_segments(hypothesis)
    speaker_labels = sorted(
        {segment["speaker_id"] for segment in reference_segments}
        | {segment["speaker_id"] for segment in hypothesis_segments}
        | {"no_overlap"}
    )
    confusion_matrix = {ref: {pred: 0 for pred in speaker_labels} for ref in speaker_labels}

    evaluated = 0
    correct = 0
    no_overlap = 0
    total_overlap_duration = 0.0
    correct_overlap_duration = 0.0
    segment_records: list[dict[str, Any]] = []

    for idx, predicted in enumerate(hypothesis_segments):
        reference_match, overlap = _best_reference_match(predicted, reference_segments)
        predicted_speaker = predicted["speaker_id"]
        reference_speaker = reference_match["speaker_id"] if reference_match else "no_overlap"
        confusion_matrix.setdefault(reference_speaker, {}).setdefault(predicted_speaker, 0)
        confusion_matrix[reference_speaker][predicted_speaker] += 1

        if reference_match is None or overlap <= 0.0:
            no_overlap += 1
            segment_records.append(_segment_record(idx, predicted, reference_speaker, overlap, False))
            continue

        evaluated += 1
        total_overlap_duration += overlap
        is_correct = predicted_speaker == reference_speaker
        if is_correct:
            correct += 1
            correct_overlap_duration += overlap
        segment_records.append(_segment_record(idx, predicted, reference_speaker, overlap, is_correct))

    result = {
        "segment_accuracy": correct / evaluated if evaluated else None,
        "duration_weighted_accuracy": correct_overlap_duration / total_overlap_duration if total_overlap_duration else None,
        "correct_segments": correct,
        "evaluated_segments": evaluated,
        "no_overlap_segments": no_overlap,
        "total_predicted_segments": len(hypothesis_segments),
        "confusion_matrix": confusion_matrix,
        "segments": segment_records,
    }
    if output_path is not None:
        save_metrics_json(result, output_path)
    return result


def evaluate_speaker_identification_batch(
    pairs: list[tuple[str | Path | dict[str, Any] | list[dict[str, Any]], str | Path | dict[str, Any] | list[dict[str, Any]]]],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate speaker-identification accuracy for multiple recordings."""
    records = [evaluate_speaker_identification(reference, hypothesis) for reference, hypothesis in pairs]
    accuracies = [record["segment_accuracy"] for record in records if record["segment_accuracy"] is not None]
    result = {
        "records": records,
        "summary": {
            "num_recordings": len(records),
            "mean_segment_accuracy": mean(accuracies) if accuracies else None,
            "total_correct_segments": sum(record["correct_segments"] for record in records),
            "total_evaluated_segments": sum(record["evaluated_segments"] for record in records),
        },
    }
    if output_path is not None:
        save_metrics_json(result, output_path)
    return result


def load_speaker_identification_results(path: str | Path) -> dict[str, Any]:
    """Load saved speaker-identification metrics."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _best_reference_match(predicted: dict[str, Any], reference_segments: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best_segment = None
    best_overlap = 0.0
    for reference in reference_segments:
        overlap = _overlap_duration(predicted, reference)
        if overlap > best_overlap:
            best_segment = reference
            best_overlap = overlap
    return best_segment, best_overlap


def _overlap_duration(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))


def _segment_record(
    idx: int,
    predicted: dict[str, Any],
    reference_speaker: str,
    overlap: float,
    correct: bool,
) -> dict[str, Any]:
    return {
        "id": predicted.get("id", predicted.get("segment_id", idx)),
        "start": predicted["start"],
        "end": predicted["end"],
        "predicted_speaker": predicted["speaker_id"],
        "reference_speaker": reference_speaker,
        "reference_overlap": overlap,
        "correct": correct,
    }
