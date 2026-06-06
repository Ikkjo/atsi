"""RTTM (Rich Transcription Time Marked) utilities for speaker diarization.

Provides read/write functions for the standard NIST RTTM format used by
diarization tools and evaluation frameworks such as pyannote.metrics.

Standard RTTM line format::

    SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Standard 10-column RTTM fields (space-delimited)
_RTTM_FIELDS = [
    "type",
    "file_id",
    "channel",
    "start",
    "duration",
    "ortho",
    "stt",
    "speaker_id",
    "stt_conf",
    "slam_conf",
]


def write_rttm(
    segments: list[dict[str, Any]],
    output_path: str | Path,
    file_id: str | None = None,
) -> Path:
    """Save diarization segments to an RTTM file.

    Args:
        segments: List of segment dicts with ``start``, ``end``, and
            ``speaker_id`` keys.  ``duration`` is optional and will be
            computed from ``end - start`` if absent.
        output_path: Destination file path.
        file_id: Optional recording identifier.  If ``None``, the stem of
            *output_path* is used.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fid = file_id if file_id is not None else output_path.stem

    lines: list[str] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        duration = float(seg["duration"]) if "duration" in seg else end - start
        speaker = str(seg["speaker_id"])
        lines.append(
            f"SPEAKER {fid} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")

    logger.info("Wrote %d RTTM lines to %s", len(lines), output_path)
    return output_path


def read_rttm(path: str | Path) -> list[dict[str, Any]]:
    """Read an RTTM file into a list of segment dictionaries.

    Args:
        path: Path to the RTTM file.

    Returns:
        List of segment dicts with ``start``, ``end``, ``duration``, and
        ``speaker_id`` keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RTTM file not found: {path}")

    segments: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                logger.warning("Skipping malformed RTTM line: %s", line)
                continue
            # parts[0] = type (SPEAKER)
            # parts[1] = file_id
            # parts[2] = channel
            # parts[3] = start
            # parts[4] = duration
            # parts[5] = ortho
            # parts[6] = stt
            # parts[7] = speaker_id
            start = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            segments.append(
                {
                    "start": start,
                    "end": start + duration,
                    "duration": duration,
                    "speaker_id": speaker_id,
                    "file_id": parts[1],
                    "channel": parts[2],
                }
            )

    return segments


def merge_adjacent_rttm_segments(
    segments: list[dict[str, Any]],
    max_gap_s: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge adjacent RTTM segments belonging to the same speaker.

    This is a post-processing helper to reduce fragmentation in the
    final transcript (see Epic 5.3).

    Args:
        segments: List of segment dicts (ordered by ``start``).
        max_gap_s: Maximum silence gap to merge across.

    Returns:
        Merged segment list.
    """
    if not segments:
        return []

    sorted_segments = sorted(segments, key=lambda s: (float(s["start"]), float(s["end"])))
    merged: list[dict[str, Any]] = [dict(sorted_segments[0])]

    for seg in sorted_segments[1:]:
        prev = merged[-1]
        gap = float(seg["start"]) - float(prev["end"])
        same_speaker = seg["speaker_id"] == prev["speaker_id"]
        if same_speaker and gap <= max_gap_s:
            prev["end"] = max(float(prev["end"]), float(seg["end"]))
            prev["duration"] = float(prev["end"]) - float(prev["start"])
        else:
            merged.append(dict(seg))

    return merged


def filter_short_rttm_segments(
    segments: list[dict[str, Any]],
    min_duration_s: float = 0.2,
) -> list[dict[str, Any]]:
    """Discard very short segments to suppress spurious VAD noise.

    See Epic 5.3 for the rationale.

    Args:
        segments: List of segment dicts.
        min_duration_s: Minimum duration to keep.

    Returns:
        Filtered segment list.
    """
    return [
        seg
        for seg in segments
        if float(seg.get("duration", float(seg["end"]) - float(seg["start"]))) >= min_duration_s
    ]
