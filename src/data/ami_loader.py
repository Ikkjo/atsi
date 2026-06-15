"""AMI Meeting Corpus loader via HuggingFace datasets."""

import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
CONFIGS = ("ihm", "sdm")
TARGET_SR = 16_000


class AMILoader:
    """Load and manage AMI Meeting Corpus data via HuggingFace datasets.

    Supports both IHM (individual headset microphone) and SDM (single
    distant microphone) configurations. Data is loaded on-demand to
    avoid loading the full corpus into RAM.

    Args:
        config: Dataset configuration ("ihm" or "sdm").
        cache_dir: Directory to cache downloaded dataset files.
    """

    def __init__(
        self,
        config: str = "ihm",
        cache_dir: str | None = None,
    ) -> None:
        if config not in CONFIGS:
            raise ValueError(f"config must be one of {CONFIGS}, got '{config}'")

        self.config = config
        self.cache_dir = cache_dir
        self._dataset: DatasetDict | None = None

    @property
    def dataset(self) -> DatasetDict:
        """Lazy-load the full dataset dictionary."""
        if self._dataset is None:
            logger.info("Loading AMI dataset (config=%s)...", self.config)
            self._dataset = load_dataset(
                "edinburghcstr/ami",
                self.config,
                cache_dir=self.cache_dir,
            )
            logger.info("AMI dataset loaded: %s", self._dataset)
        return self._dataset

    def get_split(self, split: str) -> Dataset:
        """Return a specific split (train/validation/test).

        Args:
            split: Split name.

        Returns:
            HuggingFace Dataset for the requested split.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got '{split}'")
        return self.dataset[split]

    def get_meeting_ids(self, split: str) -> list[str]:
        """Return unique meeting IDs for a given split.

        Args:
            split: Split name.

        Returns:
            Sorted list of unique meeting IDs.
        """
        ds = self.get_split(split)
        meeting_ids = sorted(set(ds["meeting_id"]))
        return meeting_ids

    def get_meeting_segments(self, meeting_id: str, split: str | None = None) -> list[dict[str, Any]]:
        """Return all segments for a specific meeting.

        Args:
            meeting_id: AMI meeting identifier (e.g. "EN2001a").
            split: Optional split to search. If None, searches all splits.

        Returns:
            List of segment dicts with keys: meeting_id, audio_id, text,
            begin_time, end_time, microphone_id, speaker_id, audio.
        """
        segments = []
        splits_to_search = [split] if split else list(SPLITS)

        for s in splits_to_search:
            ds = self.get_split(s)
            mask = [mid == meeting_id for mid in ds["meeting_id"]]
            meeting_ds = ds.filter(lambda _, i: mask[i], with_indices=True, num_proc=1)
            for row in meeting_ds:
                segments.append(row)

        if not segments:
            logger.warning("No segments found for meeting_id=%s", meeting_id)

        return segments

    def get_meeting_speakers(self, meeting_id: str, split: str | None = None) -> list[str]:
        """Return unique speaker IDs for a specific meeting.

        Args:
            meeting_id: AMI meeting identifier.
            split: Optional split to search.

        Returns:
            Sorted list of unique speaker IDs.
        """
        segments = self.get_meeting_segments(meeting_id, split)
        speakers = sorted({seg["speaker_id"] for seg in segments})
        return speakers

    def get_meeting_audio_path(self, meeting_id: str, split: str) -> str | None:
        """Return the local file path for a meeting's audio.

        Reconstructs the full meeting audio from pre-segmented utterances if
        a cached WAV does not already exist.

        Args:
            meeting_id: AMI meeting identifier.
            split: Split name.

        Returns:
            Local file path to the audio file, or None if not found.
        """
        from src.data.meeting_audio import get_meeting_audio_path

        try:
            return get_meeting_audio_path(self, meeting_id, split)
        except FileNotFoundError:
            return None

    def save_metadata(self, output_dir: str = "data/processed") -> dict[str, Any]:
        """Save dataset metadata to JSON files.

        Creates:
            - data/processed/ami_metadata.json: Overall dataset statistics
            - data/processed/ami_splits.json: Meeting IDs per split

        Args:
            output_dir: Directory to save metadata files.

        Returns:
            Metadata dictionary.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metadata: dict[str, Any] = {
            "config": self.config,
            "splits": {},
            "total_segments": 0,
        }

        split_info: dict[str, Any] = {}

        for split in SPLITS:
            ds = self.get_split(split)
            meeting_ids = self.get_meeting_ids(split)
            speakers = sorted(set(ds["speaker_id"]))

            split_info[split] = {
                "num_segments": len(ds),
                "num_meetings": len(meeting_ids),
                "meeting_ids": meeting_ids,
                "speakers": speakers,
            }

            metadata["splits"][split] = {
                "num_segments": len(ds),
                "num_meetings": len(meeting_ids),
            }
            metadata["total_segments"] += len(ds)

        metadata_path = out / "ami_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata to %s", metadata_path)

        splits_path = out / "ami_splits.json"
        with open(splits_path, "w") as f:
            json.dump(split_info, f, indent=2)
        logger.info("Saved split info to %s", splits_path)

        return metadata

    def __repr__(self) -> str:
        return f"AMILoader(config={self.config!r})"
