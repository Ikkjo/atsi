"""Validate the experiment runner on synthetic tiny cached artifacts.

This script creates minimal synthetic data (annotations, ASR JSON, embeddings)
and runs the experiment runner to verify the pipeline works end-to-end without
eeding real AMI data or GPU models.

Usage:
    .venv/bin/python tests/validation/validate_experiment_runner.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.run_experiment import run_experiment


def create_synthetic_annotations(output_dir: Path, meeting_id: str = "TEST001") -> Path:
    """Create a minimal annotation JSON."""
    annotations = {
        "meeting_id": meeting_id,
        "config": "ihm",
        "speakers": ["A", "B", "C", "D"],
        "segments": [
            {"speaker_id": "A", "begin_time": 0.0, "end_time": 5.0},
            {"speaker_id": "B", "begin_time": 5.5, "end_time": 10.0},
            {"speaker_id": "C", "begin_time": 10.5, "end_time": 15.0},
            {"speaker_id": "D", "begin_time": 15.5, "end_time": 20.0},
        ],
        "words": [
            {"word": "hello", "speaker": "A", "start": 0.5, "end": 1.0},
            {"word": "world", "speaker": "A", "start": 1.5, "end": 2.0},
            {"word": "this", "speaker": "B", "start": 6.0, "end": 6.5},
            {"word": "is", "speaker": "B", "start": 6.8, "end": 7.0},
            {"word": "a", "speaker": "C", "start": 11.0, "end": 11.2},
            {"word": "test", "speaker": "C", "start": 11.5, "end": 12.0},
            {"word": "meeting", "speaker": "D", "start": 16.0, "end": 16.5},
            {"word": "end", "speaker": "D", "start": 17.0, "end": 17.5},
        ],
        "metadata": {"word_timestamp_source": "synthetic"},
    }
    path = output_dir / f"{meeting_id}_annotations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)
    return path


def create_synthetic_asr(output_dir: Path, meeting_id: str = "TEST001") -> Path:
    """Create a minimal ASR JSON."""
    asr_result = {
        "meeting_id": meeting_id,
        "audio_path": "test.wav",
        "duration": 20.0,
        "model_id": "openai/whisper-test",
        "word_timestamp_mode": "native",
        "text": "hello world this is a test meeting end",
        "segments": [
            {"id": 0, "start": 0.5, "end": 2.0, "text": "hello world"},
            {"id": 1, "start": 6.0, "end": 7.0, "text": "this is"},
            {"id": 2, "start": 11.0, "end": 12.0, "text": "a test"},
            {"id": 3, "start": 16.0, "end": 17.5, "text": "meeting end"},
        ],
        "words": [
            {"id": 0, "start": 0.5, "end": 1.0, "word": "hello"},
            {"id": 1, "start": 1.5, "end": 2.0, "word": "world"},
            {"id": 2, "start": 6.0, "end": 6.5, "word": "this"},
            {"id": 3, "start": 6.8, "end": 7.0, "word": "is"},
            {"id": 4, "start": 11.0, "end": 11.2, "word": "a"},
            {"id": 5, "start": 11.5, "end": 12.0, "word": "test"},
            {"id": 6, "start": 16.0, "end": 16.5, "word": "meeting"},
            {"id": 7, "start": 17.0, "end": 17.5, "word": "end"},
        ],
    }
    path = output_dir / f"{meeting_id}_whisper.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asr_result, f, indent=2)
    return path


def create_synthetic_embeddings(output_dir: Path, meeting_id: str = "TEST001") -> Path:
    """Create a minimal embeddings .pt file."""
    num_segments = 4
    embedding_dim = 192
    segments = []
    for i in range(num_segments):
        segments.append({
            "segment_id": f"seg_{i:06d}",
            "start": i * 5.0,
            "end": i * 5.0 + 4.5,
            "duration": 4.5,
            "embedding_index": i,
        })

    # Create embeddings that are distinct per speaker for clustering
    embeddings = torch.zeros(num_segments, embedding_dim)
    for i in range(num_segments):
        embeddings[i, i * 10] = 1.0  # Each segment has distinct direction

    result = {
        "metadata": {
            "audio_path": "test.wav",
            "meeting_id": meeting_id,
            "model_id": "speechbrain/spkrec-ecapa-voxceleb",
            "embedding_dim": embedding_dim,
            "normalized": True,
        },
        "segments": segments,
        "embeddings": embeddings,
    }
    path = output_dir / f"{meeting_id}_ecapa_embeddings.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return path


def create_synthetic_reference_embeddings(output_dir: Path, meeting_id: str = "TEST001") -> list[Path]:
    """Create reference embeddings for scenario 3."""
    paths = []
    speakers = ["A", "B", "C", "D"]
    embedding_dim = 192

    for i, speaker in enumerate(speakers):
        embedding = torch.zeros(embedding_dim)
        embedding[i * 10] = 1.0
        data = {
            "meeting_id": meeting_id,
            "speaker_id": speaker,
            "embedding": embedding,
            "dimension": embedding_dim,
            "normalized": True,
        }
        path = output_dir / f"{meeting_id}_{speaker}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, path)
        paths.append(path)

    return paths


def create_config(
    temp_dir: Path,
    scenario: str,
    meeting_id: str = "TEST001",
) -> Path:
    """Create a minimal experiment config for validation."""
    config = {
        "experiment_id": f"{scenario}_ihm_validation",
        "scenario": scenario,
        "microphone_configuration": "ihm",
        "split": "test",
        "meeting_ids": [meeting_id],
        "seed": 42,
        "paths": {
            "annotations_dir": str(temp_dir / "annotations"),
            "asr_dir": str(temp_dir / "asr"),
            "embeddings_dir": str(temp_dir / "embeddings"),
            "reference_embeddings_dir": str(temp_dir / "references"),
            "output_dir": str(temp_dir / "output"),
        },
        "diarization": {
            "linkage_method": "average",
            "metric": "cosine",
            "distance_threshold": 0.5 if scenario == "scenario1" else None,
            "reference_threshold": None,
        },
        "integration": {
            "refine_diarization": True,
            "min_segment_duration_s": 0.2,
            "max_merge_gap_s": 0.5,
        },
        "evaluation": {
            "collar": 0.25,
            "skip_overlap": True,
            "normalize_wer": True,
        },
        "runtime": {
            "use_cached_asr": True,
            "use_cached_embeddings": True,
            "fail_fast": False,
        },
    }
    path = temp_dir / f"{scenario}_config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    return path


def validate_scenario(scenario: str, temp_dir: Path) -> bool:
    """Run validation for a single scenario."""
    print(f"\nValidating {scenario}...")
    print("=" * 60)

    meeting_id = "TEST001"

    # Create synthetic data
    create_synthetic_annotations(temp_dir / "annotations", meeting_id)
    create_synthetic_asr(temp_dir / "asr", meeting_id)
    create_synthetic_embeddings(temp_dir / "embeddings", meeting_id)
    create_synthetic_reference_embeddings(temp_dir / "references", meeting_id)

    # Create config and run
    config_path = create_config(temp_dir, scenario, meeting_id)
    result = run_experiment(config_path=config_path, dry_run=False)

    # Validate outputs
    output_dir = Path(result["output_dir"])
    success = True

    # Check required files exist
    required_files = [
        output_dir / "config.json",
        output_dir / "manifest.jsonl",
        output_dir / "diarization" / f"{meeting_id}.rttm",
        output_dir / "transcripts" / f"{meeting_id}.json",
        output_dir / "transcripts" / f"{meeting_id}.txt",
        output_dir / "metrics" / f"{meeting_id}.json",
        output_dir / "metrics" / "summary.json",
    ]

    for req_file in required_files:
        if req_file.exists():
            print(f"  ✓ {req_file.name}")
        else:
            print(f"  ✗ MISSING: {req_file.name}")
            success = False

    # Check metrics content
    metrics_path = output_dir / "metrics" / f"{meeting_id}.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        checks = [
            ("meeting_id", metrics.get("meeting_id") == meeting_id),
            ("experiment_id", "validation" in metrics.get("experiment_id", "")),
            ("scenario", metrics.get("scenario") == scenario),
            ("der", "der" in metrics and "jer" in metrics),
            ("wer", "wer" in metrics and "integrated" in metrics["wer"]),
            ("artifacts", "artifacts" in metrics),
        ]

        if scenario == "scenario3":
            checks.append(("speaker_identification", metrics.get("speaker_identification") is not None))
        else:
            checks.append(("speaker_identification_null", metrics.get("speaker_identification") is None))

        for check_name, check_result in checks:
            if check_result:
                print(f"  ✓ Metrics: {check_name}")
            else:
                print(f"  ✗ Metrics check failed: {check_name}")
                success = False

    # Check summary
    summary_path = output_dir / "metrics" / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if summary.get("num_completed") == 1:
            print(f"  ✓ Summary: 1 meeting completed")
        else:
            print(f"  ✗ Summary: expected 1 completed, got {summary.get('num_completed')}")
            success = False

    print(f"\n{'PASS' if success else 'FAIL'}: {scenario}")
    return success


def main() -> int:
    """Run validation for all three scenarios."""
    print("Experiment Runner Validation")
    print("=" * 60)
    print("Creating synthetic data and running all 3 scenarios...")

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        results = {}

        for scenario in ["scenario1", "scenario2", "scenario3"]:
            try:
                results[scenario] = validate_scenario(scenario, temp_dir)
            except Exception as exc:
                print(f"\n  ✗ EXCEPTION: {exc}")
                import traceback
                traceback.print_exc()
                results[scenario] = False

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    all_passed = all(results.values())
    for scenario, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {scenario}: {status}")

    if all_passed:
        print("\nAll validations passed!")
        return 0
    else:
        print("\nSome validations failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
