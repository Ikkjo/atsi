#!/usr/bin/env python3
"""Generate annotations for all meetings in a given AMI configuration and split.

Usage:
    .venv/bin/python scripts/generate_annotations.py --config ihm --split test
    .venv/bin/python scripts/generate_annotations.py --config sdm --split test
    .venv/bin/python scripts/generate_annotations.py --config ihm --split test --output-dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ami_loader import AMILoader
from src.data.annotation_parser import AnnotationParser


def generate_annotations(
    config: str = "ihm",
    split: str = "test",
    output_dir: str = "data/processed",
    meeting_ids: list[str] | None = None,
    cache_dir: str | None = None,
) -> list[Path]:
    """Generate annotations for all meetings in a configuration.

    Args:
        config: Microphone configuration ("ihm" or "sdm").
        split: Dataset split ("train", "validation", "test").
        output_dir: Directory to save annotation JSON files.
        meeting_ids: Optional list of specific meeting IDs. If None, processes all.

    Returns:
        List of paths to saved annotation files.
    """
    print(f"Generating annotations for {config.upper()}/{split}...")
    loader = AMILoader(config=config, cache_dir=cache_dir)
    parser = AnnotationParser(loader)

    if meeting_ids is None:
        meeting_ids = loader.get_meeting_ids(split)

    print(f"Found {len(meeting_ids)} meetings")
    paths: list[Path] = []
    failed: list[str] = []

    for idx, meeting_id in enumerate(meeting_ids, 1):
        try:
            path = parser.save_meeting_annotations(
                meeting_id, output_dir=output_dir, split=split
            )
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
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for annotations",
    )
    parser.add_argument(
        "--meeting-id",
        type=str,
        nargs="*",
        help="Specific meeting ID(s) to process. If omitted, processes all.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/media/ikkjo/StoreJet - Ilija/hf_cache",
        help="HuggingFace dataset cache directory. Defaults to external drive.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_annotations(
        config=args.config,
        split=args.split,
        output_dir=args.output_dir,
        meeting_ids=args.meeting_id,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
