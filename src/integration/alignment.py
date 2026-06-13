"""Timestamp alignment between ASR words and diarization segments.

The alignment rule intentionally stays simple and deterministic for Epic 5.1:
each word is assigned to the speaker segment with the largest temporal overlap.
If there is no overlap, or if multiple segments have exactly the same overlap,
the speaker from the segment with the nearest start time is used.
"""

from __future__ import annotations

from typing import Any


UNKNOWN_SPEAKER = "unknown"


def assign_speaker_to_word(
    word: dict[str, Any],
    diarization_segments: list[dict[str, Any]],
    unknown_speaker: str = UNKNOWN_SPEAKER,
) -> str:
    """Assign one ASR word to a speaker using overlap then nearest-start fallback.

    Args:
        word: Word dict containing ``start`` and ``end`` timestamps in seconds.
        diarization_segments: Segment dicts containing ``start``, ``end``, and
            ``speaker_id``.
        unknown_speaker: Label returned when no usable diarization segment exists.

    Returns:
        Assigned speaker ID.
    """
    if not diarization_segments:
        return unknown_speaker

    word_start = _timestamp_or_none(word.get("start"))
    word_end = _timestamp_or_none(word.get("end"))
    if word_start is None and word_end is None:
        return _nearest_segment_speaker(0.0, diarization_segments, unknown_speaker)
    if word_start is None:
        word_start = word_end
    if word_end is None:
        word_end = word_start
    assert word_start is not None
    assert word_end is not None

    if word_end < word_start:
        word_start, word_end = word_end, word_start

    scored_segments = []
    for idx, segment in enumerate(diarization_segments):
        segment_start = float(segment["start"])
        segment_end = float(segment["end"])
        overlap = max(0.0, min(word_end, segment_end) - max(word_start, segment_start))
        scored_segments.append((overlap, idx, segment))

    max_overlap = max(score[0] for score in scored_segments)
    if max_overlap <= 0.0:
        return _nearest_segment_speaker(word_start, diarization_segments, unknown_speaker)

    best = [score for score in scored_segments if score[0] == max_overlap]
    if len(best) > 1:
        return _nearest_segment_speaker(word_start, [score[2] for score in best], unknown_speaker)

    return str(best[0][2].get("speaker_id", unknown_speaker))


def align_words_to_speakers(
    words: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]],
    unknown_speaker: str = UNKNOWN_SPEAKER,
) -> list[dict[str, Any]]:
    """Return ASR words annotated with deterministic speaker assignments."""
    aligned = []
    for idx, word in enumerate(words):
        speaker_id = assign_speaker_to_word(word, diarization_segments, unknown_speaker=unknown_speaker)
        aligned_word = dict(word)
        aligned_word.setdefault("id", idx)
        aligned_word["speaker_id"] = speaker_id
        aligned.append(aligned_word)
    return aligned


def _nearest_segment_speaker(
    word_start: float,
    segments: list[dict[str, Any]],
    unknown_speaker: str,
) -> str:
    if not segments:
        return unknown_speaker
    nearest = min(
        enumerate(segments),
        key=lambda item: (abs(float(item[1]["start"]) - word_start), float(item[1]["start"]), item[0]),
    )[1]
    return str(nearest.get("speaker_id", unknown_speaker))


def _timestamp_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
