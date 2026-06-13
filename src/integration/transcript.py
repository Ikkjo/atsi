"""Final transcript generation for Epic 5.2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.integration.alignment import UNKNOWN_SPEAKER, align_words_to_speakers


def build_transcript_segments(aligned_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive speaker-labelled words into transcript segments."""
    segments: list[dict[str, Any]] = []

    for word in aligned_words:
        token = str(word.get("word", word.get("text", ""))).strip()
        if not token:
            continue

        speaker_id = str(word.get("speaker_id", UNKNOWN_SPEAKER))
        start = _timestamp_or_none(word.get("start"))
        end = _timestamp_or_none(word.get("end"))

        if segments and segments[-1]["speaker_id"] == speaker_id:
            segment = segments[-1]
            segment["text"] = _join_words(segment["text"], token)
            segment["words"].append(dict(word))
            if end is not None:
                segment["end"] = end if segment["end"] is None else max(segment["end"], end)
            continue

        segments.append(
            {
                "id": len(segments),
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "text": token,
                "words": [dict(word)],
            }
        )

    return segments


def build_integrated_transcript(
    asr_result: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    recording_name: str | None = None,
    scenario: str | None = None,
    microphone_configuration: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    unknown_speaker: str = UNKNOWN_SPEAKER,
) -> dict[str, Any]:
    """Build a JSON-serializable final transcript from ASR and diarization outputs."""
    aligned_words = align_words_to_speakers(
        asr_result.get("words") or [],
        diarization_segments,
        unknown_speaker=unknown_speaker,
    )
    transcript_segments = build_transcript_segments(aligned_words)
    inferred_recording = recording_name or asr_result.get("meeting_id") or Path(asr_result.get("audio_path", "")).stem

    metadata = {
        "recording_name": inferred_recording or None,
        "scenario": scenario,
        "microphone_configuration": microphone_configuration,
        "asr_model_id": asr_result.get("model_id"),
        "word_timestamp_mode": asr_result.get("word_timestamp_mode"),
        "duration": asr_result.get("duration"),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "metadata": metadata,
        "text": format_text_transcript(transcript_segments),
        "segments": transcript_segments,
        "words": aligned_words,
    }


def format_text_transcript(segments: list[dict[str, Any]]) -> str:
    """Format transcript segments as human-readable timestamped lines."""
    lines = []
    for segment in segments:
        start = format_timestamp(segment.get("start"))
        end = format_timestamp(segment.get("end"))
        speaker_id = segment.get("speaker_id", UNKNOWN_SPEAKER)
        text = str(segment.get("text", "")).strip()
        lines.append(f"[{start} - {end}] {speaker_id}: {text}")
    return "\n".join(lines)


def save_text_transcript(segments_or_transcript: list[dict[str, Any]] | dict[str, Any], output_path: str | Path) -> Path:
    """Save the human-readable transcript text file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(segments_or_transcript, dict):
        text = str(segments_or_transcript.get("text", ""))
    else:
        text = format_text_transcript(segments_or_transcript)
    output_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return output_path


def save_json_transcript(transcript: dict[str, Any], output_path: str | Path) -> Path:
    """Save the final integrated transcript JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(transcript), indent=2), encoding="utf-8")
    return output_path


def format_timestamp(seconds: Any) -> str:
    """Format seconds as ``HH:MM:SS`` for final text output."""
    if seconds is None:
        return "--:--:--"
    total_seconds = max(0, int(float(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _join_words(text: str, token: str) -> str:
    if token in {".", ",", "?", "!", ":", ";", "%"}:
        return f"{text}{token}"
    if token.startswith(("'", "\"")) and len(token) == 1:
        return f"{text}{token}"
    return f"{text} {token}" if text else token


def _timestamp_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value
