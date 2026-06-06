"""Parse AMI ground truth annotations into uniform JSON format.

Extracts word-level transcriptions with timestamps and speaker segments
from the HuggingFace AMI dataset.
"""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.data.ami_loader import AMILoader

logger = logging.getLogger(__name__)


class AnnotationParser:
    """Parse AMI annotations into structured JSON per meeting.

    Args:
        loader: AMILoader instance for accessing dataset.
    """

    def __init__(self, loader: AMILoader) -> None:
        self.loader = loader

    def parse_meeting(
        self,
        meeting_id: str,
        split: str | None = None,
    ) -> dict[str, Any]:
        """Parse all annotations for a single meeting.

        Args:
            meeting_id: AMI meeting identifier (e.g. "EN2001a").
            split: Optional split to search. If None, searches all splits.

        Returns:
            Dictionary with keys:
                - meeting_id: str
                - config: str (ihm/sdm)
                - speakers: list of speaker IDs
                - segments: list of speaker segment dicts
                - words: list of word-level transcription dicts
                - metadata: annotation provenance and timing notes
        """
        segments = self.loader.get_meeting_segments(meeting_id, split)
        if not segments:
            raise ValueError(f"No segments found for meeting_id={meeting_id}")

        speakers = sorted({seg["speaker_id"] for seg in segments})

        speaker_segments = self._extract_speaker_segments(segments)
        word_annotations = self._extract_word_annotations(segments)

        return {
            "meeting_id": meeting_id,
            "config": self.loader.config,
            "speakers": speakers,
            "segments": speaker_segments,
            "words": word_annotations,
            "metadata": {
                "word_timestamp_source": "estimated_from_segment_boundaries",
                "word_timestamp_note": (
                    "HuggingFace AMI rows expose segment-level begin/end times. "
                    "Word timings are estimated uniformly within each segment."
                ),
            },
        }

    def _extract_speaker_segments(self, segments: list[dict]) -> list[dict]:
        """Extract contiguous speaker segments from dataset segments.

        Merges consecutive segments from the same speaker that are close
        in time (within 0.5s gap).

        Args:
            segments: List of dataset segment dicts.

        Returns:
            List of speaker segment dicts with speaker, start, end.
        """
        sorted_segs = sorted(segments, key=lambda s: s["begin_time"])

        speaker_segments: list[dict] = []
        current_speaker: str | None = None
        current_start: float | None = None
        current_end: float | None = None

        merge_gap = 0.5

        for seg in sorted_segs:
            speaker = seg["speaker_id"]
            start = seg["begin_time"]
            end = seg["end_time"]

            if current_speaker is None:
                current_speaker = speaker
                current_start = start
                current_end = end
            elif speaker == current_speaker and (start - current_end) <= merge_gap:
                current_end = max(current_end, end)
            else:
                speaker_segments.append({
                    "speaker": current_speaker,
                    "start": round(current_start, 3),
                    "end": round(current_end, 3),
                })
                current_speaker = speaker
                current_start = start
                current_end = end

        if current_speaker is not None:
            speaker_segments.append({
                "speaker": current_speaker,
                "start": round(current_start, 3),
                "end": round(current_end, 3),
            })

        return speaker_segments

    def _extract_word_annotations(self, segments: list[dict]) -> list[dict]:
        """Extract word-level transcriptions with timestamps.

        Args:
            segments: List of dataset segment dicts.

        Returns:
            List of word annotation dicts with word, speaker, start, end.
        """
        words: list[dict] = []

        for seg in sorted(segments, key=lambda s: s["begin_time"]):
            text = seg["text"].strip()
            if not text:
                continue

            word_tokens = text.split()
            seg_duration = seg["end_time"] - seg["begin_time"]
            if len(word_tokens) == 0:
                continue

            word_duration = seg_duration / len(word_tokens)

            for i, word in enumerate(word_tokens):
                word_start = seg["begin_time"] + i * word_duration
                word_end = word_start + word_duration
                words.append({
                    "word": word,
                    "speaker": seg["speaker_id"],
                    "start": round(word_start, 3),
                    "end": round(word_end, 3),
                    "timestamp_source": "estimated_from_segment_boundaries",
                })

        return words

    def save_meeting_annotations(
        self,
        meeting_id: str,
        output_dir: str = "data/processed",
        split: str | None = None,
    ) -> Path:
        """Parse and save meeting annotations to JSON.

        Args:
            meeting_id: AMI meeting identifier.
            output_dir: Directory to save the JSON file.
            split: Optional split to search.

        Returns:
            Path to the saved JSON file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        annotations = self.parse_meeting(meeting_id, split)
        output_path = out / f"{meeting_id}_annotations.json"

        with open(output_path, "w") as f:
            json.dump(annotations, f, indent=2)

        logger.info("Saved annotations for %s to %s", meeting_id, output_path)
        return output_path

    def save_all_meetings(
        self,
        split: str,
        output_dir: str = "data/processed",
        meeting_ids: list[str] | None = None,
    ) -> list[Path]:
        """Parse and save annotations for all meetings in a split.

        Args:
            split: Split name (train/validation/test).
            output_dir: Directory to save JSON files.
            meeting_ids: Optional list of meeting IDs to process. If None,
                processes all meetings in the split.

        Returns:
            List of paths to saved JSON files.
        """
        if meeting_ids is None:
            meeting_ids = self.loader.get_meeting_ids(split)

        paths = []
        for meeting_id in meeting_ids:
            try:
                path = self.save_meeting_annotations(meeting_id, output_dir, split)
                paths.append(path)
            except Exception as e:
                logger.error("Failed to parse %s: %s", meeting_id, e)

        return paths
