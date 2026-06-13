"""WER evaluation utilities for ASR outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


def compute_wer(reference: str, hypothesis: str, normalize: bool = True) -> dict[str, Any]:
    """Compute word error rate and jiwer alignment counts."""
    from jiwer import process_words

    ref = normalize_text(reference) if normalize else reference
    hyp = normalize_text(hypothesis) if normalize else hypothesis
    output = process_words(ref, hyp)

    return {
        "wer": output.wer,
        "mer": output.mer,
        "wil": output.wil,
        "wip": output.wip,
        "hits": output.hits,
        "substitutions": output.substitutions,
        "insertions": output.insertions,
        "deletions": output.deletions,
        "reference_words": len(ref.split()),
        "hypothesis_words": len(hyp.split()),
        "normalization": "lowercase_punctuation_stripped" if normalize else "none",
    }


def evaluate_asr_wer(
    asr_output: str | Path | dict[str, Any],
    reference_annotations: str | Path | dict[str, Any],
    condition: str | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Evaluate one Whisper output JSON against one AMI annotation JSON."""
    asr_record = _load_json(asr_output)
    reference_record = _load_json(reference_annotations)

    reference_text = reference_text_from_annotations(reference_record)
    hypothesis_text = hypothesis_text_from_asr(asr_record)
    metrics = compute_wer(reference_text, hypothesis_text, normalize=normalize)

    config = reference_record.get("config")
    return {
        "meeting_id": reference_record.get("meeting_id") or asr_record.get("meeting_id"),
        "config": config,
        "condition": condition or _condition_from_config(config),
        "model_id": asr_record.get("model_id"),
        "word_timestamp_mode": asr_record.get("word_timestamp_mode"),
        "reference_text": reference_text,
        "hypothesis_text": hypothesis_text,
        **metrics,
    }


def evaluate_integrated_wer(
    integrated_transcript: str | Path | dict[str, Any],
    reference_annotations: str | Path | dict[str, Any],
    asr_output: str | Path | dict[str, Any] | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Evaluate WER for the final speaker-labelled transcript.

    If the original Whisper output is provided, the result includes both
    whisper-only and integrated WER so the diarization/integration step can be
    checked for accidental word loss or re-ordering.
    """
    integrated_record = _load_json(integrated_transcript)
    reference_record = _load_json(reference_annotations)
    reference_text = reference_text_from_annotations(reference_record)
    integrated_text = hypothesis_text_from_integrated(integrated_record)
    integrated_metrics = compute_wer(reference_text, integrated_text, normalize=normalize)

    result = {
        "meeting_id": reference_record.get("meeting_id") or integrated_record.get("metadata", {}).get("recording_name"),
        "config": reference_record.get("config") or integrated_record.get("metadata", {}).get("microphone_configuration"),
        "scenario": integrated_record.get("metadata", {}).get("scenario"),
        "reference_text": reference_text,
        "integrated_hypothesis_text": integrated_text,
        "integrated": integrated_metrics,
    }

    if asr_output is not None:
        asr_record = _load_json(asr_output)
        whisper_text = hypothesis_text_from_asr(asr_record)
        result["whisper_hypothesis_text"] = whisper_text
        result["whisper_only"] = compute_wer(reference_text, whisper_text, normalize=normalize)
        result["wer_delta_integrated_minus_whisper"] = result["integrated"]["wer"] - result["whisper_only"]["wer"]

    return result


def evaluate_integrated_wer_batch(
    pairs: list[tuple[str | Path, str | Path]],
    asr_outputs: list[str | Path] | None = None,
    output_path: str | Path | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Evaluate multiple integrated transcript/reference pairs."""
    records = []
    for idx, (integrated_path, reference_path) in enumerate(pairs):
        asr_output = asr_outputs[idx] if asr_outputs is not None else None
        records.append(evaluate_integrated_wer(integrated_path, reference_path, asr_output=asr_output, normalize=normalize))

    result = {
        "records": records,
        "summary": {
            "num_recordings": len(records),
            "integrated": _summarize_nested_wer(records, "integrated"),
            "whisper_only": _summarize_nested_wer(records, "whisper_only"),
        },
    }
    if output_path is not None:
        save_wer_results(result, output_path)
    return result


def evaluate_asr_wer_batch(
    pairs: list[tuple[str | Path, str | Path]],
    output_path: str | Path | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    """Evaluate multiple ASR/reference pairs and optionally save results."""
    records = [evaluate_asr_wer(asr_path, ref_path, normalize=normalize) for asr_path, ref_path in pairs]
    result = {
        "records": records,
        "summary": summarize_wer(records),
    }

    if output_path is not None:
        save_wer_results(result, output_path)

    return result


def match_asr_annotation_pairs(asr_dir: str | Path, annotation_dir: str | Path) -> list[tuple[Path, Path]]:
    """Match `<meeting>_whisper.json` files to `<meeting>_annotations.json` files."""
    asr_dir = Path(asr_dir)
    annotation_dir = Path(annotation_dir)
    pairs = []

    for asr_path in sorted(asr_dir.glob("*_whisper.json")):
        meeting_id = asr_path.name.removesuffix("_whisper.json")
        annotation_path = annotation_dir / f"{meeting_id}_annotations.json"
        if annotation_path.exists():
            pairs.append((asr_path, annotation_path))

    return pairs


def summarize_wer(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate WER overall and by microphone configuration/condition."""
    return {
        "num_recordings": len(records),
        "overall": _summarize_group(records),
        "by_config": _group_summary(records, "config"),
        "by_condition": _group_summary(records, "condition"),
    }


def save_wer_results(results: dict[str, Any], output_path: str | Path) -> Path:
    """Save WER evaluation results as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    return output_path


def hypothesis_text_from_asr(asr_record: dict[str, Any]) -> str:
    """Extract hypothesis text from normalized Whisper output."""
    text = (asr_record.get("text") or "").strip()
    if text:
        return text
    segments = asr_record.get("segments") or []
    text = " ".join((segment.get("text") or "").strip() for segment in segments).strip()
    if text:
        return text
    words = asr_record.get("words") or []
    return " ".join((word.get("word") or word.get("text") or "").strip() for word in words).strip()


def hypothesis_text_from_integrated(integrated_record: dict[str, Any]) -> str:
    """Extract hypothesis words from the final integrated transcript JSON."""
    words = integrated_record.get("words") or []
    if words:
        return " ".join((word.get("word") or word.get("text") or "").strip() for word in words).strip()

    segments = integrated_record.get("segments") or []
    return " ".join((segment.get("text") or "").strip() for segment in segments).strip()


def reference_text_from_annotations(annotation_record: dict[str, Any]) -> str:
    """Extract reference text from AMI annotations."""
    words = annotation_record.get("words") or []
    if words:
        return " ".join((word.get("word") or "").strip() for word in words).strip()

    segments = annotation_record.get("segments") or []
    return " ".join((segment.get("text") or "").strip() for segment in segments).strip()


def normalize_text(text: str) -> str:
    """Normalize text before WER calculation."""
    text = text.lower()
    text = re.sub(r"[^\w\s']+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_json(value: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    with open(value) as f:
        return json.load(f)


def _condition_from_config(config: Any) -> str | None:
    if config == "ihm":
        return "clean_headset"
    if config == "sdm":
        return "distant_mic"
    return None


def _group_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group_key = str(record.get(key) or "unknown")
        groups.setdefault(group_key, []).append(record)
    return {group_key: _summarize_group(group_records) for group_key, group_records in groups.items()}


def _summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_recordings": 0, "mean_wer": None, "total_reference_words": 0}
    return {
        "num_recordings": len(records),
        "mean_wer": mean(record["wer"] for record in records),
        "total_reference_words": sum(record["reference_words"] for record in records),
        "total_hypothesis_words": sum(record["hypothesis_words"] for record in records),
        "total_substitutions": sum(record["substitutions"] for record in records),
        "total_insertions": sum(record["insertions"] for record in records),
        "total_deletions": sum(record["deletions"] for record in records),
    }


def _summarize_nested_wer(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    nested = [record[key] for record in records if key in record]
    if not nested:
        return {"num_recordings": 0, "mean_wer": None, "total_reference_words": 0}
    return {
        "num_recordings": len(nested),
        "mean_wer": mean(record["wer"] for record in nested),
        "total_reference_words": sum(record["reference_words"] for record in nested),
        "total_hypothesis_words": sum(record["hypothesis_words"] for record in nested),
        "total_substitutions": sum(record["substitutions"] for record in nested),
        "total_insertions": sum(record["insertions"] for record in nested),
        "total_deletions": sum(record["deletions"] for record in nested),
    }
