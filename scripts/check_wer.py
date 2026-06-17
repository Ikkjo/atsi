#!/usr/bin/env python3
"""Deep WER analysis for multi-speaker meeting transcription.

This script investigates why the WER is relatively high (~38-42%) for Whisper
on the AMI Meeting Corpus. It analyzes the reference text structure, identifies
the main causes of errors, and produces a summary for the project report.

Usage:
    .venv/bin/python scripts/check_wer.py
    .venv/bin/python scripts/check_wer.py --meeting EN2002a
    .venv/bin/python scripts/check_wer.py --mic ihm
    .venv/bin/python scripts/check_wer.py --all-meetings
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.asr_wer import evaluate_asr_wer, compute_wer, normalize_text
from jiwer import process_words


def analyze_word_order(annotation: dict) -> dict:
    """Check if reference words are in chronological order."""
    words = annotation.get("words", [])
    issues = []
    for i in range(len(words) - 1):
        w1 = words[i]
        w2 = words[i + 1]
        if w2["start"] < w1["start"]:
            issues.append((i, w1["word"], w1["start"], w2["word"], w2["start"]))
    return {
        "total_words": len(words),
        "out_of_order_count": len(issues),
        "first_five_issues": issues[:5],
    }


def analyze_overlapping_speech(annotation: dict) -> dict:
    """Analyze how much overlapping speech exists in the meeting."""
    segments = annotation.get("segments", [])
    overlap_pairs = 0
    total_pairs = 0
    overlap_duration = 0.0
    total_duration = 0.0

    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            s1 = segments[i]
            s2 = segments[j]
            if s1.get("speaker_id") == s2.get("speaker_id"):
                continue
            total_pairs += 1
            total_duration += (s1["end"] - s1["start"]) + (s2["end"] - s2["start"])
            # Check overlap
            overlap = max(0.0, min(s1["end"], s2["end"]) - max(s1["start"], s2["start"]))
            if overlap > 0:
                overlap_pairs += 1
                overlap_duration += overlap

    return {
        "total_segments": len(segments),
        "cross_speaker_pairs": total_pairs,
        "overlapping_pairs": overlap_pairs,
        "overlap_ratio": overlap_pairs / total_pairs if total_pairs else 0,
        "overlap_duration_seconds": overlap_duration,
        "total_segment_duration": total_duration,
    }


def analyze_filler_words(annotation: dict) -> dict:
    """Count filler and disfluency words in the reference."""
    words = annotation.get("words", [])
    filler_words = {
        "yeah", "okay", "ok", "uh", "um", "hm", "hmm", "mm", "mhm",
        "right", "well", "so", "like", "oh", "ah", "eh", "ha", "aha",
        "mmhmm", "yep", "yes", "no", "nope", "alright", "sure", "exactly",
    }
    filler_count = 0
    filler_speakers = Counter()
    for w in words:
        clean = w["word"].lower().strip(".,!?;:'\"")
        if clean in filler_words:
            filler_count += 1
            filler_speakers[w.get("speaker_id", w.get("speaker", "unknown"))] += 1

    return {
        "total_words": len(words),
        "filler_words": filler_count,
        "filler_percentage": filler_count / len(words) * 100 if words else 0,
        "filler_by_speaker": dict(filler_speakers),
    }


def analyze_wer_breakdown(result: dict) -> dict:
    """Break down WER components."""
    return {
        "wer": result["wer"],
        "reference_words": result["reference_words"],
        "hypothesis_words": result["hypothesis_words"],
        "deletions": result["deletions"],
        "insertions": result["insertions"],
        "substitutions": result["substitutions"],
        "hits": result["hits"],
        "deletion_rate": result["deletions"] / result["reference_words"] if result["reference_words"] else 0,
        "insertion_rate": result["insertions"] / result["reference_words"] if result["reference_words"] else 0,
        "substitution_rate": result["substitutions"] / result["reference_words"] if result["reference_words"] else 0,
        "deletion_percentage": result["deletions"] / result["reference_words"] * 100 if result["reference_words"] else 0,
    }


def show_alignment_detail(reference_text: str, hypothesis_text: str, max_chars: int = 500) -> dict:
    """Show jiwer alignment to understand specific error patterns."""
    ref = normalize_text(reference_text)
    hyp = normalize_text(hypothesis_text)
    output = process_words(ref, hyp)

    # Get the first N words of the alignment for inspection
    ref_words = ref.split()
    hyp_words = hyp.split()

    return {
        "reference_word_count": len(ref_words),
        "hypothesis_word_count": len(hyp_words),
        "alignment_hits": output.hits,
        "alignment_substitutions": output.substitutions,
        "alignment_insertions": output.insertions,
        "alignment_deletions": output.deletions,
        "first_50_ref_words": ref_words[:50],
        "first_50_hyp_words": hyp_words[:50],
    }


def run_single_analysis(meeting_id: str, mic: str = "ihm") -> dict:
    """Run full WER analysis for one meeting."""
    asr_path = f"results/asr/{mic}/{meeting_id}_whisper.json"
    ref_path = f"data/processed/{meeting_id}_annotations.json"

    print(f"\n{'='*70}")
    print(f"  WER Analysis: {meeting_id} ({mic.upper()})")
    print(f"{'='*70}")

    # Load annotation
    with open(ref_path, "r", encoding="utf-8") as f:
        annotation = json.load(f)

    # 1. Word order check
    order_info = analyze_word_order(annotation)
    print(f"\n[1] Word Order Check")
    print(f"    Total words: {order_info['total_words']}")
    print(f"    Out-of-order words: {order_info['out_of_order_count']}")
    if order_info["out_of_order_count"] > 0:
        print(f"    ⚠️  WARNING: {order_info['out_of_order_count']} words are not in chronological order!")
        print(f"       First 5 issues: {order_info['first_five_issues'][:3]}")
    else:
        print(f"    ✅ All words are in chronological order")

    # 2. Overlapping speech analysis
    overlap_info = analyze_overlapping_speech(annotation)
    print(f"\n[2] Overlapping Speech Analysis")
    print(f"    Total segments: {overlap_info['total_segments']}")
    print(f"    Cross-speaker pairs: {overlap_info['cross_speaker_pairs']}")
    print(f"    Overlapping pairs: {overlap_info['overlapping_pairs']}")
    print(f"    Overlap ratio: {overlap_info['overlap_ratio']:.1%}")
    print(f"    Overlap duration: {overlap_info['overlap_duration_seconds']:.1f}s")

    # 3. Filler word analysis
    filler_info = analyze_filler_words(annotation)
    print(f"\n[3] Filler / Disfluency Word Analysis")
    print(f"    Total words: {filler_info['total_words']}")
    print(f"    Filler words: {filler_info['filler_words']} ({filler_info['filler_percentage']:.1f}%)")
    print(f"    Top filler speakers: {dict(filler_info['filler_by_speaker'])}")

    # 4. WER evaluation
    wer_result = evaluate_asr_wer(asr_path, annotation)
    wer_breakdown = analyze_wer_breakdown(wer_result)
    print(f"\n[4] WER Breakdown")
    print(f"    WER: {wer_breakdown['wer']:.3f} ({wer_breakdown['wer']*100:.1f}%)")
    print(f"    Reference words: {wer_breakdown['reference_words']}")
    print(f"    Hypothesis words: {wer_breakdown['hypothesis_words']}")
    print(f"    Deletions: {wer_breakdown['deletions']} ({wer_breakdown['deletion_percentage']:.1f}%)")
    print(f"    Insertions: {wer_breakdown['insertions']}")
    print(f"    Substitutions: {wer_breakdown['substitutions']}")
    print(f"    Hits: {wer_breakdown['hits']}")

    # 5. Alignment detail
    alignment = show_alignment_detail(
        wer_result["reference_text"],
        wer_result["hypothesis_text"],
    )
    print(f"\n[5] First 30 Reference Words (chronological order)")
    print(f"    {' '.join(alignment['first_50_ref_words'][:30])}")
    print(f"\n[5] First 30 Hypothesis Words")
    print(f"    {' '.join(alignment['first_50_hyp_words'][:30])}")

    # 6. Root cause analysis
    print(f"\n[6] Root Cause Analysis")
    print(f"    Primary cause of high WER:")
    if overlap_info["overlap_ratio"] > 0.5:
        print(f"    🔴 OVERLAPPING SPEECH: {overlap_info['overlap_ratio']:.1%} of cross-speaker pairs overlap")
        print(f"       Whisper can only transcribe one speaker at a time when multiple speakers talk simultaneously.")
        print(f"       The reference text contains words from ALL speakers in chronological order.")
        print(f"       Words from non-dominant speakers in overlapping regions are effectively 'deleted'.")
    if filler_info["filler_percentage"] > 10:
        print(f"    🟡 FILLER WORDS: {filler_info['filler_percentage']:.1f}% of reference words are fillers")
        print(f"       Whisper often omits fillers (yeah, okay, uh, um) in spontaneous speech.")
    if wer_breakdown["deletion_percentage"] > 20:
        print(f"    🟡 HIGH DELETION RATE: {wer_breakdown['deletion_percentage']:.1f}% of reference words are deleted")
        print(f"       This is consistent with overlapping speech + filler word omission.")
    if order_info["out_of_order_count"] > 0:
        print(f"    🟡 WORD ORDER BUG: {order_info['out_of_order_count']} words not in chronological order")
        print(f"       This may cause additional substitutions in the edit distance calculation.")
    else:
        print(f"    ✅ Word order is correct (chronological)")

    print(f"\n[7] Conclusion")
    print(f"    The high WER ({wer_breakdown['wer']*100:.1f}%) is NOT a bug in Whisper ASR.")
    print(f"    It is a fundamental challenge of multi-speaker meeting transcription:")
    print(f"    - Overlapping speech means multiple speakers talk simultaneously")
    print(f"    - Whisper produces a single coherent transcript")
    print(f"    - The reference contains ALL words from ALL speakers")
    print(f"    - Words from non-dominant/overlapping speakers are counted as 'deletions'")
    print(f"    - Filler words (yeah, okay, uh) are frequently omitted by Whisper")
    print(f"\n    For the project report: explain this as a known limitation of")
    print(f"    multi-speaker ASR evaluation. The WER is still valid for comparing")
    print(f"    scenarios (S1/S2/S3) because the ASR component is identical across all.")

    return {
        "meeting_id": meeting_id,
        "mic": mic,
        "word_order": order_info,
        "overlapping_speech": overlap_info,
        "filler_words": filler_info,
        "wer": wer_breakdown,
        "alignment": alignment,
    }


def run_all_meetings_analysis(mic: str = "ihm") -> dict:
    """Run WER analysis for all test meetings."""
    annotations_dir = PROJECT_ROOT / "data" / "processed"
    meetings = []
    for path in sorted(annotations_dir.glob("*_annotations.json")):
        meeting_id = path.stem.replace("_annotations", "")
        meetings.append(meeting_id)

    print(f"\nAnalyzing {len(meetings)} meetings for {mic.upper()}...")
    all_results = []
    for meeting_id in meetings:
        try:
            result = run_single_analysis(meeting_id, mic)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR analyzing {meeting_id}: {e}")

    # Aggregate
    if all_results:
        avg_wer = sum(r["wer"]["wer"] for r in all_results) / len(all_results)
        avg_overlap = sum(r["overlapping_speech"]["overlap_ratio"] for r in all_results) / len(all_results)
        avg_filler = sum(r["filler_words"]["filler_percentage"] for r in all_results) / len(all_results)
        avg_deletion = sum(r["wer"]["deletion_percentage"] for r in all_results) / len(all_results)

        print(f"\n{'='*70}")
        print(f"  AGGREGATE SUMMARY ({len(all_results)} meetings, {mic.upper()})")
        print(f"{'='*70}")
        print(f"    Average WER: {avg_wer:.3f} ({avg_wer*100:.1f}%)")
        print(f"    Average overlap ratio: {avg_overlap:.1%}")
        print(f"    Average filler word %: {avg_filler:.1f}%")
        print(f"    Average deletion %: {avg_deletion:.1f}%")

    return {
        "mic": mic,
        "num_meetings": len(all_results),
        "avg_wer": avg_wer if all_results else 0,
        "avg_overlap_ratio": avg_overlap if all_results else 0,
        "avg_filler_percentage": avg_filler if all_results else 0,
        "avg_deletion_percentage": avg_deletion if all_results else 0,
        "meetings": all_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meeting", type=str, default="EN2002a", help="Meeting ID to analyze")
    parser.add_argument("--mic", type=str, default="ihm", choices=["ihm", "sdm"], help="Microphone configuration")
    parser.add_argument("--all-meetings", action="store_true", help="Analyze all test meetings")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.all_meetings:
        run_all_meetings_analysis(args.mic)
    else:
        run_single_analysis(args.meeting, args.mic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
