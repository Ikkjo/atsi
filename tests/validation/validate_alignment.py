"""Validate cached ASR + RTTM integration outputs.

If paths are provided, this script validates the real cached-output path. Without
paths it runs a tiny synthetic validation so the check stays usable before large
AMI artifacts are available locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diarization.rttm import read_rttm  # noqa: E402
from src.integration.transcript import build_integrated_transcript, save_json_transcript, save_text_transcript  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-json", type=Path)
    parser.add_argument("--rttm", type=Path)
    parser.add_argument("--recording-name", default="validation_meeting")
    parser.add_argument("--scenario", default="validation")
    parser.add_argument("--mic", default="ihm")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "transcripts" / "validation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    asr_result, diarization_segments = _load_inputs(args)
    transcript = build_integrated_transcript(
        asr_result,
        diarization_segments,
        recording_name=args.recording_name,
        scenario=args.scenario,
        microphone_configuration=args.mic,
    )

    _validate_transcript(transcript)
    json_path = save_json_transcript(transcript, args.output_dir / f"{args.recording_name}.json")
    text_path = save_text_transcript(transcript, args.output_dir / f"{args.recording_name}.txt")
    print(f"PASS: alignment validation wrote {json_path} and {text_path}")
    return 0


def _load_inputs(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    if bool(args.asr_json) != bool(args.rttm):
        raise ValueError("Pass both --asr-json and --rttm, or neither for synthetic validation")

    if args.asr_json and args.rttm:
        with open(args.asr_json, encoding="utf-8") as f:
            return json.load(f), read_rttm(args.rttm)

    return (
        {
            "meeting_id": args.recording_name,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.1, "end": 0.4},
                {"word": "world", "start": 0.6, "end": 0.9},
            ],
            "model_id": "synthetic",
        },
        [
            {"start": 0.0, "end": 0.5, "speaker_id": "Speaker_A"},
            {"start": 0.55, "end": 1.0, "speaker_id": "Speaker_B"},
        ],
    )


def _validate_transcript(transcript: dict) -> None:
    json.dumps(transcript)
    assert transcript["words"], "expected at least one aligned word"
    assert transcript["segments"], "expected at least one transcript segment"
    assert transcript["text"].strip(), "expected non-empty text transcript"

    for word in transcript["words"]:
        assert word.get("speaker_id"), "word is missing speaker_id"
        if word.get("start") is not None and word.get("end") is not None:
            assert float(word["start"]) <= float(word["end"]), "word timestamps are not ordered"

    for segment in transcript["segments"]:
        assert segment.get("speaker_id"), "segment is missing speaker_id"
        if segment.get("start") is not None and segment.get("end") is not None:
            assert float(segment["start"]) <= float(segment["end"]), "segment timestamps are not ordered"


if __name__ == "__main__":
    raise SystemExit(main())
