"""Official AMI train/validation/test split management.

Uses the standard split provided by the HuggingFace AMI dataset.
Saves split metadata as JSON for traceability.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.data.ami_loader import AMILoader

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


class SplitManager:
    """Manage and persist AMI dataset splits.

    Args:
        loader: AMILoader instance for accessing dataset.
    """

    def __init__(self, loader: AMILoader) -> None:
        self.loader = loader

    def get_split_metadata(self) -> dict[str, Any]:
        """Generate metadata for all splits.

        Returns:
            Dictionary with split names as keys and metadata as values.
        """
        metadata: dict[str, Any] = {
            "config": self.loader.config,
            "splits": {},
        }

        for split in SPLITS:
            meeting_ids = self.loader.get_meeting_ids(split)
            speakers = set()

            ds = self.loader.get_split(split)
            for speaker_id in ds["speaker_id"]:
                speakers.add(speaker_id)

            metadata["splits"][split] = {
                "num_meetings": len(meeting_ids),
                "num_segments": len(ds),
                "meeting_ids": meeting_ids,
                "unique_speakers": sorted(speakers),
            }

        return metadata

    def save_splits(
        self,
        output_dir: str = "data/processed",
        filename: str = "ami_splits.json",
    ) -> Path:
        """Save split metadata to a JSON file.

        Args:
            output_dir: Directory to save the file.
            filename: Name of the output file.

        Returns:
            Path to the saved JSON file.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metadata = self.get_split_metadata()
        output_path = out / filename

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Saved split metadata to %s", output_path)
        return output_path

    def load_splits(self, path: str | Path) -> dict[str, Any]:
        """Load split metadata from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Split metadata dictionary.
        """
        with open(path) as f:
            return json.load(f)

    def get_meeting_split(self, meeting_id: str) -> str | None:
        """Determine which split a meeting belongs to.

        Args:
            meeting_id: AMI meeting identifier.

        Returns:
            Split name, or None if not found.
        """
        for split in SPLITS:
            if meeting_id in self.loader.get_meeting_ids(split):
                return split
        return None

    def verify_consistency(self) -> bool:
        """Verify that splits are mutually exclusive and complete.

        Returns:
            True if splits are consistent, False otherwise.
        """
        all_meeting_ids: set[str] = set()

        for split in SPLITS:
            meeting_ids = set(self.loader.get_meeting_ids(split))

            overlap = all_meeting_ids & meeting_ids
            if overlap:
                logger.error("Overlap detected in split '%s': %s", split, overlap)
                return False

            all_meeting_ids |= meeting_ids

        if not all_meeting_ids:
            logger.error("No meetings found in any split")
            return False

        logger.info(
            "Split consistency verified: %d unique meetings across %d splits",
            len(all_meeting_ids),
            len(SPLITS),
        )
        return True
