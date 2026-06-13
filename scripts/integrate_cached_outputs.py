"""Build final transcripts from cached Whisper JSON and diarization RTTM files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diarization.rttm import read_rttm  # noqa: E402
from src.integration.transcript import build_integrated_transcript, save_json_transcript, save_text_transcript  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asr-json", required=True, type=Path, help="Cached Whisper JSON path")
    parser.add_argument("--rttm", required=True, type=Path, help="Predicted diarization RTTM path")
    parser.add_argument("--recording-name", required=True, help="Recording/meeting identifier")
    parser.add_argument("--scenario", required=True, help="Scenario identifier, e.g. scenario1")
    parser.add_argument("--mic", required=True, help="Microphone configuration, e.g. ihm or sdm")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "transcripts")
    parser.add_argument("--min-segment-duration-s", type=float, default=0.2)
    parser.add_argument("--max-merge-gap-s", type=float, default=0.5)
    parser.add_argument("--no-refine-diarization", action="store_true", help="Disable Epic 5.3 diarization refinement")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.asr_json, encoding="utf-8") as f:
        asr_result = json.load(f)

    diarization_segments = read_rttm(args.rttm)
    transcript = build_integrated_transcript(
        asr_result,
        diarization_segments,
        recording_name=args.recording_name,
        scenario=args.scenario,
        microphone_configuration=args.mic,
        refine_diarization=not args.no_refine_diarization,
        min_segment_duration_s=args.min_segment_duration_s,
        max_merge_gap_s=args.max_merge_gap_s,
        extra_metadata={
            "asr_json": str(args.asr_json),
            "rttm": str(args.rttm),
        },
    )

    output_stem = f"{args.recording_name}_{args.mic}_{args.scenario}"
    save_json_transcript(transcript, args.output_dir / f"{output_stem}.json")
    save_text_transcript(transcript, args.output_dir / f"{output_stem}.txt")
    print(f"Wrote transcript outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
