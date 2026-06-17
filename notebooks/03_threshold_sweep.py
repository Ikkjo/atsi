#!/usr/bin/env python3
"""Threshold sweep for Scenario 1 on the validation set.

This script finds the optimal AHC distance threshold for Scenario 1 (unknown number
of speakers) by sweeping candidate values on the validation split of AMI Meeting Corpus.

Usage:
    .venv/bin/python notebooks/03_threshold_sweep.py
    .venv/bin/python notebooks/03_threshold_sweep.py --mic ihm
    .venv/bin/python notebooks/03_threshold_sweep.py --thresholds 0.3 0.35 0.4 0.45 0.5
    .venv/bin/python notebooks/03_threshold_sweep.py --update-configs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ami_loader import AMILoader
from src.data.meeting_audio import get_meeting_audio_path
from src.diarization.embeddings import ECAPAEmbeddingExtractor
from src.diarization.segmentation import DiarizationSegmenter, EmbeddingSegmentationConfig
from src.diarization.vad import PyannoteVAD, PyannoteVADConfig
from src.diarization.clustering import sweep_threshold
from src.utils.hardware import get_device


def discover_validation_meetings(annotations_dir: Path) -> list[str]:
    """Discover meeting IDs from validation annotation files."""
    if not annotations_dir.exists():
        return []
    meetings = []
    for path in sorted(annotations_dir.glob("*_annotations.json")):
        meeting_id = path.stem.replace("_annotations", "")
        meetings.append(meeting_id)
    return meetings


def load_validation_annotations(meeting_ids: list[str], annotations_dir: Path) -> dict[str, dict]:
    """Load reference annotations for validation meetings."""
    annotations = {}
    for meeting_id in meeting_ids:
        path = annotations_dir / f"{meeting_id}_annotations.json"
        if not path.exists():
            print(f"WARNING: Annotation not found for {meeting_id}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        segments = []
        for seg in data.get("segments", []):
            speaker_id = seg.get("speaker_id", seg.get("speaker", "unknown"))
            start = seg.get("begin_time", seg.get("start", 0))
            end = seg.get("end_time", seg.get("end", 0))
            segments.append({
                "speaker_id": str(speaker_id),
                "begin_time": float(start),
                "end_time": float(end),
                "start": float(start),
                "end": float(end),
            })
        annotations[meeting_id] = {
            "meeting_id": meeting_id,
            "speakers": data.get("speakers", []),
            "segments": segments,
        }
    return annotations


def reconstruct_validation_audio(meeting_ids: list[str], mic: str, split: str, hf_cache_dir: str) -> dict[str, str]:
    """Reconstruct audio for all validation meetings."""
    loader = AMILoader(config=mic, cache_dir=hf_cache_dir)
    audio_paths = {}
    failed = []
    for idx, meeting_id in enumerate(meeting_ids, 1):
        try:
            path = get_meeting_audio_path(loader, meeting_id, split)
            audio_paths[meeting_id] = path
            print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: {path}")
        except Exception as e:
            failed.append(meeting_id)
            print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: FAILED - {e}")
    print(f"\nAudio reconstruction: {len(audio_paths)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed}")
    return audio_paths


def generate_validation_embeddings(
    meeting_ids: list[str],
    mic: str,
    audio_paths: dict[str, str],
    output_dir: Path,
    vad_enabled: bool = True,
) -> dict[str, dict]:
    """Generate ECAPA embeddings for validation meetings."""
    import torch
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = DiarizationSegmenter(
        vad=PyannoteVAD(PyannoteVADConfig(enabled=vad_enabled)),
        config=EmbeddingSegmentationConfig(),
    )
    extractor = ECAPAEmbeddingExtractor(segmenter=segmenter)
    results = {}
    for idx, meeting_id in enumerate(meeting_ids, 1):
        audio_path = audio_paths.get(meeting_id)
        if not audio_path:
            print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: SKIPPED (no audio)")
            continue
        cache_path = output_dir / f"{meeting_id}_ecapa_embeddings.pt"
        if cache_path.exists():
            print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: Loading cached embeddings")
            results[meeting_id] = torch.load(cache_path, map_location="cpu", weights_only=False)
            continue
        print(f"  [{idx}/{len(meeting_ids)}] {meeting_id}: Extracting embeddings...")
        try:
            result = extractor.extract_embeddings(
                audio_path,
                meeting_id=meeting_id,
                output_dir=output_dir,
                use_cache=False,
            )
            results[meeting_id] = result
            print(f"    -> {len(result['segments'])} segments, dim={result['embeddings'].shape[1]}")
        except Exception as e:
            print(f"    -> FAILED: {e}")
    return results


def prepare_sweep_inputs(
    embeddings_dict: dict[str, dict],
    annotations_dict: dict[str, dict],
) -> tuple[list, list, list]:
    """Prepare aligned lists for sweep_threshold."""
    embeddings_list = []
    segments_list = []
    reference_segments_list = []
    common_meetings = sorted(set(embeddings_dict.keys()) & set(annotations_dict.keys()))
    for meeting_id in common_meetings:
        emb_result = embeddings_dict[meeting_id]
        ann = annotations_dict[meeting_id]
        embeddings = emb_result["embeddings"]
        segments = emb_result["segments"]
        ref_segments = ann["segments"]
        if len(embeddings) == 0 or len(segments) == 0:
            print(f"WARNING: {meeting_id} has no embeddings or segments, skipping")
            continue
        if len(embeddings) != len(segments):
            print(f"WARNING: {meeting_id} embedding/segment count mismatch: {len(embeddings)} vs {len(segments)}")
            continue
        embeddings_list.append(embeddings.cpu().numpy())
        segments_list.append(segments)
        reference_segments_list.append(ref_segments)
    print(f"\nPrepared {len(embeddings_list)} meetings for sweep")
    return embeddings_list, segments_list, reference_segments_list


def run_sweep_for_mic(
    mic: str,
    meeting_ids: list[str],
    annotations_dict: dict[str, dict],
    hf_cache_dir: str,
    thresholds: list[float],
    linkage_method: str = "average",
    metric: str = "cosine",
    split: str = "validation",
    vad_enabled: bool = True,
) -> dict:
    """Run the full pipeline: audio reconstruction -> embeddings -> sweep for one mic config."""
    print(f"\n{'='*60}")
    print(f"Processing {mic.upper()}")
    print(f"{'='*60}")

    # Step 1: Reconstruct audio
    print(f"\n[1/3] Reconstructing audio for {len(meeting_ids)} meetings...")
    audio_paths = reconstruct_validation_audio(meeting_ids, mic, split, hf_cache_dir)

    # Step 2: Generate embeddings
    embeddings_dir = PROJECT_ROOT / "results" / "embeddings_validation" / mic
    print(f"\n[2/3] Generating embeddings...")
    embeddings_dict = generate_validation_embeddings(
        meeting_ids, mic, audio_paths, embeddings_dir, vad_enabled=vad_enabled
    )

    # Step 3: Run sweep
    print(f"\n[3/3] Running threshold sweep...")
    embeddings_list, segments_list, ref_segments_list = prepare_sweep_inputs(
        embeddings_dict, annotations_dict
    )
    if len(embeddings_list) == 0:
        print(f"ERROR: No valid meetings for {mic}, cannot run sweep")
        return {}

    best_threshold, results = sweep_threshold(
        embeddings_list,
        segments_list,
        ref_segments_list,
        thresholds=thresholds,
        linkage_method=linkage_method,
        metric=metric,
    )

    print(f"\n{'='*60}")
    print(f"RESULTS for {mic.upper()}")
    print(f"{'='*60}")
    print(f"Best threshold: {best_threshold}")
    print(f"Best pseudo-DER: {results['best_pseudo_der']:.4f}")
    print(f"\n{'Threshold':>12} {'Avg Pseudo-DER':>18} {'# Meetings':>12}")
    print("-" * 50)
    for detail in results["sweep"]:
        t = detail["threshold"]
        der = detail["avg_pseudo_der"]
        n = len(detail["per_meeting_der"])
        marker = " <-- BEST" if t == best_threshold else ""
        print(f"{t:>12.2f} {der:>18.4f} {n:>12}{marker}")

    return {
        "best_threshold": float(best_threshold),
        "best_pseudo_der": float(results["best_pseudo_der"]),
        "results": results,
    }


def update_config(config_path: Path, new_threshold: float) -> None:
    """Update a Scenario 1 config file with the validated threshold."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    old_threshold = config["diarization"].get("distance_threshold")
    config["diarization"]["distance_threshold"] = new_threshold
    if "note" in config["diarization"]:
        del config["diarization"]["note"]
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Updated {config_path.name}: {old_threshold} -> {new_threshold}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mic", type=str, choices=["ihm", "sdm"], help="Run sweep for only one mic config")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
                        help="Candidate distance thresholds to evaluate")
    parser.add_argument("--hf-cache-dir", type=str, default="/media/ikkjo/StoreJet - Ilija/hf_cache",
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--annotations-dir", type=Path, default=PROJECT_ROOT / "data" / "validation",
                        help="Directory containing validation annotations")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use")
    parser.add_argument("--linkage-method", type=str, default="average", help="AHC linkage method")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric")
    parser.add_argument("--vad-enabled", action="store_true", default=True, help="Enable pyannote VAD")
    parser.add_argument("--update-configs", action="store_true", help="Update Scenario 1 configs with best thresholds")
    parser.add_argument("--save-json", type=Path, default=PROJECT_ROOT / "results" / "threshold_sweep_validation.json",
                        help="Path to save sweep results JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Device: {get_device()}")
    print(f"Validation annotations: {args.annotations_dir}")
    print(f"Candidate thresholds: {args.thresholds}")

    # Discover meetings
    meeting_ids = discover_validation_meetings(args.annotations_dir)
    print(f"\nFound {len(meeting_ids)} validation meetings")
    for m in meeting_ids:
        print(f"  {m}")

    # Load annotations
    annotations = load_validation_annotations(meeting_ids, args.annotations_dir)
    print(f"\nLoaded annotations for {len(annotations)} meetings")

    # Determine which mic configs to run
    mic_configs = [args.mic] if args.mic else ["ihm", "sdm"]

    # Run sweep for each mic
    all_results = {}
    for mic in mic_configs:
        result = run_sweep_for_mic(
            mic=mic,
            meeting_ids=meeting_ids,
            annotations_dict=annotations,
            hf_cache_dir=args.hf_cache_dir,
            thresholds=args.thresholds,
            linkage_method=args.linkage_method,
            metric=args.metric,
            split=args.split,
            vad_enabled=args.vad_enabled,
        )
        all_results[mic] = result

    # Save results
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "timestamp": str(np.datetime64("now")),
            "split": args.split,
            "num_meetings": len(meeting_ids),
            "candidate_thresholds": [float(t) for t in args.thresholds],
            "linkage_method": args.linkage_method,
            "metric": args.metric,
            "results": {},
        }
        for mic in mic_configs:
            if not all_results[mic]:
                continue
            results = all_results[mic]["results"]
            sweep_details = []
            for detail in results["sweep"]:
                sweep_details.append({
                    "threshold": float(detail["threshold"]),
                    "avg_pseudo_der": float(detail["avg_pseudo_der"]),
                    "per_meeting_der": [float(d) for d in detail["per_meeting_der"]],
                })
            output_data["results"][mic] = {
                "best_threshold": all_results[mic]["best_threshold"],
                "best_pseudo_der": all_results[mic]["best_pseudo_der"],
                "sweep": sweep_details,
            }
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved sweep results to: {args.save_json}")

    # Update configs
    if args.update_configs:
        print(f"\n{'='*60}")
        print("Updating Scenario 1 configs")
        print(f"{'='*60}")
        config_dir = PROJECT_ROOT / "experiments" / "configs"
        for mic in mic_configs:
            if not all_results.get(mic):
                continue
            config_path = config_dir / f"scenario1_{mic}.json"
            best_t = all_results[mic]["best_threshold"]
            update_config(config_path, best_t)
        print(f"\nNext steps:")
        print(f"  .venv/bin/python experiments/run_experiment.py --config experiments/configs/scenario1_ihm.json")
        print(f"  .venv/bin/python experiments/run_experiment.py --config experiments/configs/scenario1_sdm.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
