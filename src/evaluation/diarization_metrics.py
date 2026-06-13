"""DER and JER evaluation utilities for diarization outputs."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from src.diarization.rttm import read_rttm

DEFAULT_COLLAR_S = 0.25
DEFAULT_SKIP_OVERLAP = True


def compute_der(
    reference: str | Path | dict[str, Any] | list[dict[str, Any]],
    hypothesis: str | Path | dict[str, Any] | list[dict[str, Any]],
    collar: float = DEFAULT_COLLAR_S,
    skip_overlap: bool = DEFAULT_SKIP_OVERLAP,
    uri: str | None = None,
) -> dict[str, Any]:
    """Compute Diarization Error Rate using pyannote.metrics."""
    from pyannote.metrics.diarization import DiarizationErrorRate

    reference_segments = load_diarization_segments(reference)
    hypothesis_segments = load_diarization_segments(hypothesis)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    reference_annotation = segments_to_annotation(reference_segments, uri=uri)
    hypothesis_annotation = segments_to_annotation(hypothesis_segments, uri=uri)

    der = float(metric(reference_annotation, hypothesis_annotation))
    components = {str(key): float(value) for key, value in metric.compute_components(reference_annotation, hypothesis_annotation).items()}
    total = components.get("total", 0.0)

    return {
        "der": der,
        "collar": collar,
        "skip_overlap": skip_overlap,
        "components": {
            "missed_speech": components.get("missed detection", 0.0),
            "false_alarm": components.get("false alarm", 0.0),
            "speaker_confusion": components.get("confusion", 0.0),
            "correct": components.get("correct", 0.0),
            "total": total,
        },
        "component_rates": _component_rates(components, total),
        "num_reference_segments": len(reference_segments),
        "num_hypothesis_segments": len(hypothesis_segments),
    }


def compute_jer(
    reference: str | Path | dict[str, Any] | list[dict[str, Any]],
    hypothesis: str | Path | dict[str, Any] | list[dict[str, Any]],
    collar: float = DEFAULT_COLLAR_S,
    skip_overlap: bool = DEFAULT_SKIP_OVERLAP,
    uri: str | None = None,
) -> dict[str, Any]:
    """Compute Jaccard Error Rate using pyannote.metrics."""
    from pyannote.metrics.diarization import JaccardErrorRate

    reference_segments = load_diarization_segments(reference)
    hypothesis_segments = load_diarization_segments(hypothesis)
    metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)

    jer = float(
        metric(
            segments_to_annotation(reference_segments, uri=uri),
            segments_to_annotation(hypothesis_segments, uri=uri),
        )
    )
    return {
        "jer": jer,
        "collar": collar,
        "skip_overlap": skip_overlap,
        "num_reference_segments": len(reference_segments),
        "num_hypothesis_segments": len(hypothesis_segments),
        "interpretation": "JER weights speakers more evenly than DER, so speakers with less speech affect the score more.",
    }


def evaluate_diarization_metrics(
    reference: str | Path | dict[str, Any] | list[dict[str, Any]],
    hypothesis: str | Path | dict[str, Any] | list[dict[str, Any]],
    collar: float = DEFAULT_COLLAR_S,
    skip_overlap: bool = DEFAULT_SKIP_OVERLAP,
    uri: str | None = None,
) -> dict[str, Any]:
    """Compute DER and JER for one recording."""
    return {
        "der": compute_der(reference, hypothesis, collar=collar, skip_overlap=skip_overlap, uri=uri),
        "jer": compute_jer(reference, hypothesis, collar=collar, skip_overlap=skip_overlap, uri=uri),
    }


def evaluate_diarization_metrics_batch(
    pairs: list[tuple[str | Path | dict[str, Any] | list[dict[str, Any]], str | Path | dict[str, Any] | list[dict[str, Any]]]],
    collar: float = DEFAULT_COLLAR_S,
    skip_overlap: bool = DEFAULT_SKIP_OVERLAP,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate DER/JER for multiple reference/hypothesis pairs."""
    records = [
        evaluate_diarization_metrics(reference, hypothesis, collar=collar, skip_overlap=skip_overlap)
        for reference, hypothesis in pairs
    ]
    result = {
        "records": records,
        "summary": {
            "num_recordings": len(records),
            "mean_der": mean(record["der"]["der"] for record in records) if records else None,
            "mean_jer": mean(record["jer"]["jer"] for record in records) if records else None,
        },
    }
    if output_path is not None:
        save_metrics_json(result, output_path)
    return result


def load_diarization_segments(value: str | Path | dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load diarization segments from RTTM path, annotation JSON, or list."""
    if isinstance(value, list):
        return [_normalize_segment(segment) for segment in value]
    if isinstance(value, dict):
        return [_normalize_segment(segment) for segment in value.get("segments", [])]

    path = Path(value)
    if path.suffix.lower() == ".rttm":
        return [_normalize_segment(segment) for segment in read_rttm(path)]
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return load_diarization_segments(data)


def segments_to_annotation(segments: list[dict[str, Any]], uri: str | None = None):
    """Convert project segment dicts into a pyannote.core.Annotation."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for idx, segment in enumerate(segments):
        start = float(segment["start"])
        end = float(segment["end"])
        if end <= start:
            continue
        annotation[Segment(start, end), f"track_{idx}"] = str(segment["speaker_id"])
    return annotation


def save_metrics_json(results: dict[str, Any], output_path: str | Path) -> Path:
    """Save metric results to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return output_path


def _normalize_segment(segment: dict[str, Any]) -> dict[str, Any]:
    speaker_id = segment.get("speaker_id", segment.get("speaker", "unknown"))
    start = float(segment.get("start", segment.get("begin_time")))
    if "end" in segment:
        end = float(segment["end"])
    else:
        end = float(segment["end_time"])
    return {
        **segment,
        "start": start,
        "end": end,
        "duration": float(segment.get("duration", end - start)),
        "speaker_id": str(speaker_id),
    }


def _component_rates(components: dict[str, float], total: float) -> dict[str, float | None]:
    if total <= 0.0:
        return {"missed_speech": None, "false_alarm": None, "speaker_confusion": None}
    return {
        "missed_speech": components.get("missed detection", 0.0) / total,
        "false_alarm": components.get("false alarm", 0.0) / total,
        "speaker_confusion": components.get("confusion", 0.0) / total,
    }
