#!/usr/bin/env python3
"""Reconstruct full meeting audio from HuggingFace dataset segments.

The AMI dataset stores audio as pre-segmented utterances. This script
reconstructs full meeting audio by grouping segments by microphone,
sorting by time, and placing each clip at its correct offset.

Usage:
    .venv/bin/python scripts/reconstruct_audio.py --config ihm --split test
    .venv/bin/python scripts/reconstruct_audio.py --config sdm --split test
    .venv/bin/python scripts/reconstruct_audio.py --config ihm --split test --meeting-id EN2002a
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import _get_meeting_audio_path
from src.data.ami_loader import AMILoader


def reconstruct_audio(
    config: str = "ihm",
    split: str = "test",
    meeting_ids: list[str] | None = None,
) -> list[str]:
    """Reconstruct audio for all meetings in a configuration.

    Args:
        config: Microphone configuration ("ihm" or "sdm").
        split: Dataset split ("train", "validation", "test").
        meeting_ids: Optional list of specific meeting IDs. If None, processes all.

    Returns:
        List of paths to reconstructed audio files.
    """
    print(f"Reconstructing audio for {config.upper()}/{split}...")
    loader = AMILoader(config=config)

    if meeting_ids is None:
        meeting_ids = loader.get_meeting_ids(split)

    print(f"Found {len(meeting_ids)} meetings")
    paths: list[str] = []
    failed: list[str] = []

    experiment_config = {
        "microphone_configuration": config,
        "split": split,
    }

    for idx, meeting_id in enumerate(meeting_ids, 1):
        try:
            path = _get_meeting_audio_path(experiment_config, meeting_id)
            paths.append(path)
            print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: {path}")
        except Exception as e:
            failed.append(meeting_id)
            print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: FAILED - {e}")

    print(f"\nComplete: {len(paths)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed meetings: {failed}")

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="ihm",
        choices=["ihm", "sdm"],
        help="Microphone configuration",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--meeting-id",
        type=str,
        nargs="*",
        help="Specific meeting ID(s) to process. If omitted, processes all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reconstruct_audio(
        config=args.config,
        split=args.split,
        meeting_ids=args.meeting_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
