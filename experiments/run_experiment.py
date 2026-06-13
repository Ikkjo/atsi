"""Config-driven experiment runner for Epic 6.5.

Runs the full pipeline (ASR -> diarization -> integration -> evaluation) for a single
experiment configuration. Supports both cached artifacts and on-the-fly generation.

Usage:
    .venv/bin/python experiments/run_experiment.py \
        --config experiments/configs/scenario2_ihm.json

    .venv/bin/python experiments/run_experiment.py \
        --config experiments/configs/scenario2_ihm.json --dry-run

    .venv/bin/python experiments/run_experiment.py \
        --config experiments/configs/scenario2_ihm.json --meeting-id EN2001a

    .venv/bin/python experiments/run_experiment.py \
        --config experiments/configs/scenario2_ihm.json --no-cache-asr --no-cache-embeddings
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diarization.embeddings import ECAPAEmbeddingExtractor
from src.diarization.rttm import write_rttm
from src.diarization.scenarios import (
    run_scenario1_unknown_speakers,
    run_scenario2_oracle_speakers,
    run_scenario3_reference_identification,
)
from src.diarization.segmentation import DiarizationSegmenter
from src.integration.transcript import (
    build_integrated_transcript,
    save_json_transcript,
    save_text_transcript,
)
from src.evaluation.asr_wer import evaluate_integrated_wer
from src.evaluation.diarization_metrics import evaluate_diarization_metrics
from src.evaluation.speaker_identification import evaluate_speaker_identification
from src.asr.whisper import WhisperASR
from src.data.annotation_parser import AnnotationParser
from src.data.ami_loader import AMILoader
from src.data.preprocessing import load_audio
from src.data.reference_embeddings import ReferenceEmbeddingExtractor
from src.utils.hardware import get_device
from src.utils.logging import setup_logger
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading / validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = {
    "experiment_id",
    "scenario",
    "microphone_configuration",
    "paths",
    "diarization",
    "integration",
    "evaluation",
    "runtime",
}

# Optional fields that are recognised but not required
OPTIONAL_FIELDS = {"seed", "meeting_ids", "hf_cache_dir", "split"}

VALID_SCENARIOS = {"scenario1", "scenario2", "scenario3"}
VALID_MICS = {"ihm", "sdm"}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment config."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required fields and values."""
    missing = REQUIRED_FIELDS - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config fields: {sorted(missing)}")

    scenario = config.get("scenario")
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Invalid scenario: {scenario}. Must be one of {VALID_SCENARIOS}")

    mic = config.get("microphone_configuration")
    if mic not in VALID_MICS:
        raise ValueError(f"Invalid microphone_configuration: {mic}. Must be one of {VALID_MICS}")

    paths = config.get("paths", {})
    for key in ("annotations_dir", "asr_dir", "embeddings_dir", "output_dir"):
        if key not in paths:
            raise ValueError(f"Missing required path key: paths.{key}")

    if scenario == "scenario3":
        if "reference_embeddings_dir" not in paths:
            raise ValueError("Scenario 3 requires paths.reference_embeddings_dir")

    diarization = config.get("diarization", {})
    if "linkage_method" not in diarization:
        raise ValueError("Missing diarization.linkage_method")
    if "metric" not in diarization:
        raise ValueError("Missing diarization.metric")


def resolve_config_paths(config: dict[str, Any], project_root: Path | None = None) -> dict[str, Any]:
    """Convert relative paths to absolute paths in a copy of the config."""
    if project_root is None:
        project_root = PROJECT_ROOT
    resolved = dict(config)
    paths = dict(config.get("paths", {}))
    for key, value in paths.items():
        if value is not None:
            paths[key] = str(Path(value).resolve())
    resolved["paths"] = paths
    return resolved


# ---------------------------------------------------------------------------
# Meeting discovery
# ---------------------------------------------------------------------------

def discover_meetings(config: dict[str, Any]) -> list[str]:
    """Discover meeting IDs from config or annotation directory."""
    meeting_ids = config.get("meeting_ids")
    if meeting_ids is not None:
        return list(meeting_ids)

    annotations_dir = Path(config["paths"]["annotations_dir"])
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")

    meetings = []
    for path in sorted(annotations_dir.glob("*_annotations.json")):
        meetings.append(path.stem.removesuffix("_annotations"))

    if not meetings:
        raise ValueError(f"No *_annotations.json files found in {annotations_dir}")

    return meetings


# ---------------------------------------------------------------------------
# Per-meeting helpers
# ---------------------------------------------------------------------------

def _load_annotations(config: dict[str, Any], meeting_id: str) -> dict[str, Any]:
    """Load reference annotations for a meeting."""
    annotations_dir = Path(config["paths"]["annotations_dir"])
    path = annotations_dir / f"{meeting_id}_annotations.json"
    if not path.exists():
        raise FileNotFoundError(f"Annotations not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_meeting_audio_path(config: dict[str, Any], meeting_id: str) -> str:
    """Resolve audio path, materialising full meeting audio from HuggingFace dataset segments.

    The AMI dataset is loaded through HuggingFace ``datasets``.  Recent
    versions (4.8+) use ``torchcodec`` and store audio as pre-segmented
    utterances (one audio clip per row).  This helper reconstructs the
    full meeting audio by grouping segments by microphone channel, sorting
    by time, and placing each clip at its correct offset in a single WAV.
    """
    import torchaudio
    import torch
    from collections import defaultdict

    mic = config["microphone_configuration"]
    split = config.get("split", "test")
    cache_dir = config.get("hf_cache_dir")

    # 1. Re-use an already-materialised WAV file
    raw_dir = PROJECT_ROOT / "data" / "raw" / mic
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_path = raw_dir / f"{meeting_id}.wav"
    if wav_path.exists():
        return str(wav_path)

    # 2. Load from HuggingFace dataset
    loader = AMILoader(config=mic, cache_dir=cache_dir)
    segments = loader.get_meeting_segments(meeting_id, split)
    if not segments:
        raise FileNotFoundError(
            f"No segments found for {meeting_id} in AMI {mic}/{split}. "
            f"Ensure the HuggingFace dataset is downloaded (cache_dir={cache_dir})."
        )

    # 3. Group segments by microphone channel and sort by time
    channels: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        channels[seg["microphone_id"]].append(seg)
    for ch in channels:
        channels[ch].sort(key=lambda s: s["begin_time"])

    # 4. Determine full meeting duration
    max_end = max(seg["end_time"] for seg in segments)
    sample_rate = 16000
    total_samples = int(max_end * sample_rate) + 1

    # 5. Build multi-channel audio (one channel per microphone)
    # For IHM: typically 4 channels (H00, H01, H02, H03)
    # For SDM: typically 1 channel
    channel_ids = sorted(channels.keys())
    num_channels = len(channel_ids)
    full_audio = torch.zeros(num_channels, total_samples, dtype=torch.float32)

    for ch_idx, ch_id in enumerate(channel_ids):
        for seg in channels[ch_id]:
            # Decode audio clip
            audio_decoder = seg["audio"]
            samples = audio_decoder.get_all_samples()
            clip = samples.data  # shape (1, clip_samples)
            clip_sr = getattr(samples, "sample_rate", 16000)

            # Resample if needed (should be 16kHz already)
            if clip_sr != sample_rate:
                clip = torchaudio.functional.resample(clip, clip_sr, sample_rate)

            # Place at correct offset
            start_sample = int(seg["begin_time"] * sample_rate)
            clip_len = clip.shape[1]
            end_sample = min(start_sample + clip_len, total_samples)
            actual_len = end_sample - start_sample
            if actual_len > 0:
                full_audio[ch_idx, start_sample:end_sample] = clip[0, :actual_len]

    # 6. Convert to mono for downstream compatibility (Whisper/ECAPA expect mono)
    # For IHM, we could keep multi-channel, but mono is safer for compatibility
    if num_channels > 1:
        mono_audio = full_audio.mean(dim=0, keepdim=True)
    else:
        mono_audio = full_audio

    torchaudio.save(str(wav_path), mono_audio, sample_rate)
    return str(wav_path)


def _load_or_generate_asr(
    config: dict[str, Any], meeting_id: str, logger: logging.Logger
) -> dict[str, Any]:
    """Load cached ASR or generate it if missing and allowed."""
    asr_dir = Path(config["paths"]["asr_dir"])
    asr_path = asr_dir / f"{meeting_id}_whisper.json"
    use_cached = config["runtime"].get("use_cached_asr", True)

    if use_cached and asr_path.exists():
        logger.info("Using cached ASR: %s", asr_path)
        with open(asr_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if not use_cached:
        logger.info("Generating ASR for %s (use_cached_asr=False)", meeting_id)
        audio_path = _get_meeting_audio_path(config, meeting_id)
        logger.info("Audio path resolved: %s", audio_path)

        whisper = WhisperASR()
        result = whisper.transcribe(audio_path, meeting_id=meeting_id, output_dir=asr_dir)
        return result

    raise FileNotFoundError(
        f"Cached ASR not found: {asr_path}. "
        f"Set runtime.use_cached_asr=false to generate on-the-fly."
    )


def _load_or_generate_embeddings(
    config: dict[str, Any], meeting_id: str, logger: logging.Logger
) -> dict[str, Any]:
    """Load cached embeddings or generate them if missing and allowed."""
    embeddings_dir = Path(config["paths"]["embeddings_dir"])
    emb_path = embeddings_dir / f"{meeting_id}_ecapa_embeddings.pt"
    use_cached = config["runtime"].get("use_cached_embeddings", True)

    if use_cached and emb_path.exists():
        logger.info("Using cached embeddings: %s", emb_path)
        import torch
        return torch.load(emb_path, map_location="cpu", weights_only=False)

    if not use_cached:
        logger.info("Generating embeddings for %s (use_cached_embeddings=False)", meeting_id)
        audio_path = _get_meeting_audio_path(config, meeting_id)
        logger.info("Audio path resolved: %s", audio_path)

        extractor = ECAPAEmbeddingExtractor()
        result = extractor.extract_embeddings(
            audio_path, meeting_id=meeting_id, output_dir=embeddings_dir, use_cache=False
        )
        return result

    raise FileNotFoundError(
        f"Cached embeddings not found: {emb_path}. "
        f"Set runtime.use_cached_embeddings=false to generate on-the-fly."
    )


def _load_or_generate_references(
    config: dict[str, Any], meeting_id: str, logger: logging.Logger
) -> dict[str, Any]:
    """Load reference embeddings for scenario 3, auto-generating if missing."""
    ref_dir = Path(config["paths"]["reference_embeddings_dir"])
    annotation = _load_annotations(config, meeting_id)
    speakers = annotation.get("speakers", [])

    references: dict[str, Any] = {}
    missing_speakers = []

    for speaker_id in speakers:
        ref_path = ref_dir / f"{meeting_id}_{speaker_id}.pt"
        if ref_path.exists():
            import torch
            data = torch.load(ref_path, map_location="cpu", weights_only=False)
            references[speaker_id] = data["embedding"]
        else:
            missing_speakers.append(speaker_id)

    if missing_speakers:
        logger.info(
            "Auto-generating reference embeddings for %s / speakers: %s",
            meeting_id, missing_speakers
        )
        # Need to generate via ReferenceEmbeddingExtractor
        # This requires an AMI loader for the meeting audio
        mic = config["microphone_configuration"]
        cache_dir = config.get("hf_cache_dir")
        loader = AMILoader(config=mic, cache_dir=cache_dir)
        extractor = ReferenceEmbeddingExtractor(loader)
        for speaker_id in missing_speakers:
            try:
                extractor.save_reference_embedding(
                    meeting_id, speaker_id, split=config.get("split", "test"), output_dir=str(ref_dir)
                )
                import torch
                data = torch.load(ref_dir / f"{meeting_id}_{speaker_id}.pt", map_location="cpu", weights_only=False)
                references[speaker_id] = data["embedding"]
            except Exception as exc:
                logger.warning("Failed to generate reference for %s/%s: %s", meeting_id, speaker_id, exc)
                raise

    if not references:
        raise ValueError(f"No reference embeddings found or generated for {meeting_id}")

    return references


# ---------------------------------------------------------------------------
# Per-meeting processing
# ---------------------------------------------------------------------------

def run_one_meeting(
    config: dict[str, Any],
    meeting_id: str,
    output_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Process a single meeting end-to-end.

    Returns:
        Dict with keys: metrics_path, metrics, status, elapsed_seconds.
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Processing meeting: %s", meeting_id)
    logger.info("=" * 60)

    if dry_run:
        logger.info("[DRY-RUN] Would process meeting: %s", meeting_id)
        return {
            "metrics_path": None,
            "metrics": None,
            "status": "dry_run",
            "elapsed_seconds": 0.0,
        }

    # Load reference annotations
    annotations = _load_annotations(config, meeting_id)
    logger.info("Loaded annotations for %s (speakers: %s)", meeting_id, annotations.get("speakers"))

    # Load/generate ASR
    asr_result = _load_or_generate_asr(config, meeting_id, logger)
    logger.info("Loaded ASR for %s (words: %d)", meeting_id, len(asr_result.get("words", [])))

    # Load/generate embeddings
    emb_result = _load_or_generate_embeddings(config, meeting_id, logger)
    segments = emb_result["segments"]
    embeddings = emb_result["embeddings"]
    logger.info(
        "Loaded embeddings for %s (segments: %d, dim: %s)",
        meeting_id, len(segments), embeddings.shape[1] if embeddings.ndim == 2 else 0
    )

    # Run scenario
    scenario = config["scenario"]
    diarization_kwargs: dict[str, Any] = {}

    if scenario == "scenario1":
        threshold = config["diarization"].get("distance_threshold")
        if threshold is None:
            logger.warning("distance_threshold is None for scenario1; skipping %s", meeting_id)
            raise ValueError(
                "distance_threshold is required for scenario1. "
                "Run a validation sweep to select a threshold."
            )
        diarization_kwargs = {
            "segments": segments,
            "embeddings": embeddings,
            "reference_segments": annotations.get("segments", []),
            "distance_threshold": threshold,
            "linkage_method": config["diarization"]["linkage_method"],
            "metric": config["diarization"]["metric"],
        }
        result = run_scenario1_unknown_speakers(**diarization_kwargs)

    elif scenario == "scenario2":
        n_speakers = len(annotations.get("speakers", []))
        if n_speakers <= 0:
            raise ValueError(f"No speakers found in annotations for {meeting_id}")
        diarization_kwargs = {
            "segments": segments,
            "embeddings": embeddings,
            "reference_segments": annotations.get("segments", []),
            "n_speakers": n_speakers,
            "linkage_method": config["diarization"]["linkage_method"],
            "metric": config["diarization"]["metric"],
        }
        result = run_scenario2_oracle_speakers(**diarization_kwargs)

    elif scenario == "scenario3":
        references = _load_or_generate_references(config, meeting_id, logger)
        logger.info("Loaded %d reference embeddings for %s", len(references), meeting_id)
        diarization_kwargs = {
            "segments": segments,
            "embeddings": embeddings,
            "references": references,
            "threshold": config["diarization"].get("reference_threshold"),
        }
        result = run_scenario3_reference_identification(**diarization_kwargs)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    diarization_segments = result["diarization_segments"]
    logger.info("Diarization complete: %d segments", len(diarization_segments))

    # Save RTTM
    rttm_path = output_dir / "diarization" / f"{meeting_id}.rttm"
    write_rttm(diarization_segments, rttm_path, file_id=meeting_id)
    logger.info("Saved RTTM: %s", rttm_path)

    # Build integrated transcript
    transcript = build_integrated_transcript(
        asr_result,
        diarization_segments,
        recording_name=meeting_id,
        scenario=scenario,
        microphone_configuration=config["microphone_configuration"],
        refine_diarization=config["integration"].get("refine_diarization", True),
        min_segment_duration_s=config["integration"].get("min_segment_duration_s", 0.2),
        max_merge_gap_s=config["integration"].get("max_merge_gap_s", 0.5),
    )
    logger.info("Built integrated transcript: %d words, %d segments", len(transcript.get("words", [])), len(transcript.get("segments", [])))

    # Save transcripts
    transcript_json_path = output_dir / "transcripts" / f"{meeting_id}.json"
    transcript_txt_path = output_dir / "transcripts" / f"{meeting_id}.txt"
    save_json_transcript(transcript, transcript_json_path)
    save_text_transcript(transcript, transcript_txt_path)
    logger.info("Saved transcripts: %s", transcript_json_path)

    # Compute DER / JER
    diarization_metrics = evaluate_diarization_metrics(
        annotations,
        diarization_segments,
        collar=config["evaluation"].get("collar", 0.25),
        skip_overlap=config["evaluation"].get("skip_overlap", True),
    )
    logger.info(
        "DER=%.3f JER=%.3f (missed=%.3f false=%.3f conf=%.3f)",
        diarization_metrics["der"]["der"],
        diarization_metrics["jer"]["jer"],
        diarization_metrics["der"]["components"]["missed_speech"],
        diarization_metrics["der"]["components"]["false_alarm"],
        diarization_metrics["der"]["components"]["speaker_confusion"],
    )

    # Compute WER
    wer_result = evaluate_integrated_wer(
        transcript,
        annotations,
        asr_output=asr_result,
        normalize=config["evaluation"].get("normalize_wer", True),
    )
    logger.info(
        "WER integrated=%.3f whisper-only=%.3f delta=%.3f",
        wer_result["integrated"]["wer"],
        wer_result.get("whisper_only", {}).get("wer", 0.0),
        wer_result.get("wer_delta_integrated_minus_whisper", 0.0),
    )

    # Speaker identification (scenario 3)
    speaker_id_result = None
    if scenario == "scenario3":
        speaker_id_result = evaluate_speaker_identification(
            annotations,
            diarization_segments,
        )
        logger.info(
            "Speaker identification accuracy: %.3f (segment) %.3f (duration-weighted)",
            speaker_id_result.get("segment_accuracy", 0.0) or 0.0,
            speaker_id_result.get("duration_weighted_accuracy", 0.0) or 0.0,
        )

    # Build metrics record
    metrics: dict[str, Any] = {
        "meeting_id": meeting_id,
        "experiment_id": config["experiment_id"],
        "scenario": scenario,
        "microphone_configuration": config["microphone_configuration"],
        "der": diarization_metrics["der"],
        "jer": diarization_metrics["jer"],
        "wer": {
            "integrated": wer_result["integrated"],
            "whisper_only": wer_result.get("whisper_only"),
            "wer_delta_integrated_minus_whisper": wer_result.get("wer_delta_integrated_minus_whisper"),
        },
        "speaker_identification": speaker_id_result,
        "artifacts": {
            "rttm": str(rttm_path),
            "transcript_json": str(transcript_json_path),
            "transcript_txt": str(transcript_txt_path),
        },
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    # Save per-meeting metrics
    metrics_path = output_dir / "metrics" / f"{meeting_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info("Completed meeting %s in %.1fs", meeting_id, metrics["elapsed_seconds"])

    return {
        "metrics_path": str(metrics_path),
        "metrics": metrics,
        "status": "completed",
        "elapsed_seconds": metrics["elapsed_seconds"],
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_metrics(output_dir: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-meeting metrics into a summary."""
    from statistics import mean, stdev

    completed = [m for m in manifest if m["status"] == "completed"]
    failed = [m for m in manifest if m["status"] == "failed"]
    skipped = [m for m in manifest if m["status"] == "skipped"]

    ders = []
    jers = []
    wers_integrated = []
    wers_whisper = []
    speaker_accs = []
    elapsed_times = []

    for record in completed:
        metrics_path = record.get("metrics_file")
        if not metrics_path:
            continue
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            continue

        ders.append(metrics["der"]["der"])
        jers.append(metrics["jer"]["jer"])
        wers_integrated.append(metrics["wer"]["integrated"]["wer"])
        if metrics["wer"].get("whisper_only"):
            wers_whisper.append(metrics["wer"]["whisper_only"]["wer"])
        elapsed_times.append(metrics.get("elapsed_seconds", 0))
        if metrics.get("speaker_identification"):
            acc = metrics["speaker_identification"].get("segment_accuracy")
            if acc is not None:
                speaker_accs.append(acc)

    def _stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
        return {
            "count": len(values),
            "mean": round(mean(values), 4),
            "std": round(stdev(values), 4) if len(values) > 1 else 0.0,
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    return {
        "experiment_id": config["experiment_id"] if "config" in dir() else "unknown",
        "num_meetings_total": len(manifest),
        "num_completed": len(completed),
        "num_failed": len(failed),
        "num_skipped": len(skipped),
        "der": _stats(ders),
        "jer": _stats(jers),
        "wer_integrated": _stats(wers_integrated),
        "wer_whisper_only": _stats(wers_whisper),
        "speaker_identification_accuracy": _stats(speaker_accs),
        "elapsed_seconds": _stats(elapsed_times),
        "failed_meetings": [m["meeting_id"] for m in failed],
        "skipped_meetings": [m["meeting_id"] for m in skipped],
    }


def _save_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    summary_path = output_dir / "metrics" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary: %s", summary_path)


def _save_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for record in manifest:
            f.write(json.dumps(record, default=str) + "\n")
    logger.info("Saved manifest: %s", manifest_path)


# ---------------------------------------------------------------------------
# Main experiment orchestration
# ---------------------------------------------------------------------------

def run_experiment(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    meeting_id: str | None = None,
    dry_run: bool = False,
    fail_fast: bool = False,
    use_cached_asr: bool | None = None,
    use_cached_embeddings: bool | None = None,
) -> dict[str, Any]:
    """Run a full experiment from a config file.

    Returns:
        Dict with keys: output_dir, summary, manifest, experiment_id.
    """
    config = load_config(config_path)
    validate_config(config)
    config = resolve_config_paths(config)

    # Apply CLI overrides to runtime config
    if use_cached_asr is not None:
        config.setdefault("runtime", {})["use_cached_asr"] = use_cached_asr
    if use_cached_embeddings is not None:
        config.setdefault("runtime", {})["use_cached_embeddings"] = use_cached_embeddings

    # Set random seed
    set_seed(config.get("seed", 42))

    # Create timestamped output directory
    experiment_id = config["experiment_id"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        base_output = Path(config["paths"]["output_dir"])
        out_dir = base_output.parent / f"{base_output.name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    exp_logger = setup_logger(
        name="experiment",
        log_dir=str(out_dir / "logs"),
        experiment_id=experiment_id,
        level=logging.INFO,
    )
    exp_logger.info("Experiment started: %s", experiment_id)
    exp_logger.info("Config path: %s", Path(config_path).resolve())
    exp_logger.info("Output directory: %s", out_dir.resolve())

    # Save resolved config
    config_save_path = out_dir / "config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)
    exp_logger.info("Saved resolved config: %s", config_save_path)

    # Create subdirectories
    for subdir in ("diarization", "transcripts", "metrics", "logs"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Discover meetings
    if dry_run:
        try:
            meetings = discover_meetings(config)
            exp_logger.info("[DRY-RUN] Discovered %d meetings: %s", len(meetings), meetings)
        except Exception as exc:
            exp_logger.error("[DRY-RUN] Failed to discover meetings: %s", exc)
            raise
        exp_logger.info("[DRY-RUN] Validation complete. Would process %d meetings.", len(meetings))
        return {
            "output_dir": str(out_dir),
            "summary": {"dry_run": True, "num_meetings_discovered": len(meetings) if "meetings" in dir() else 0},
            "manifest": [],
            "experiment_id": experiment_id,
        }

    meetings = discover_meetings(config)
    if meeting_id is not None:
        if meeting_id not in meetings:
            raise ValueError(f"Meeting {meeting_id} not found in discovered meetings: {meetings}")
        meetings = [meeting_id]
        exp_logger.info("Override: processing only meeting %s", meeting_id)

    exp_logger.info("Discovered %d meetings: %s", len(meetings), meetings)

    # Process each meeting
    manifest: list[dict[str, Any]] = []
    for mid in meetings:
        try:
            result = run_one_meeting(config, mid, out_dir, exp_logger, dry_run=False)
            manifest.append({
                "meeting_id": mid,
                "status": result["status"],
                "metrics_file": result["metrics_path"],
                "elapsed_seconds": result["elapsed_seconds"],
                "timestamp": datetime.now().isoformat(),
            })
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            exp_logger.error("Failed to process %s: %s", mid, error_msg)
            exp_logger.debug("Traceback: %s", traceback.format_exc())
            manifest.append({
                "meeting_id": mid,
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
            })
            if fail_fast or config.get("runtime", {}).get("fail_fast", False):
                exp_logger.error("Fail-fast enabled. Aborting experiment.")
                break

    # Aggregate
    summary = _aggregate_metrics(out_dir, manifest)
    summary["experiment_id"] = experiment_id
    summary["config_path"] = str(Path(config_path).resolve())
    summary["output_dir"] = str(out_dir.resolve())
    summary["timestamp"] = datetime.now().isoformat()

    _save_summary(out_dir, summary)
    _save_manifest(out_dir, manifest)

    exp_logger.info("Experiment complete: %d/%d meetings succeeded", summary["num_completed"], summary["num_meetings_total"])
    exp_logger.info("Mean DER: %s", summary["der"]["mean"])
    exp_logger.info("Mean WER (integrated): %s", summary["wer_integrated"]["mean"])

    return {
        "output_dir": str(out_dir),
        "summary": summary,
        "manifest": manifest,
        "experiment_id": experiment_id,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to experiment config JSON")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--meeting-id", type=str, help="Process only this meeting")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and file discovery without running models")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed meeting")
    parser.add_argument("--no-cache-asr", action="store_true", help="Ignore cached ASR and re-run Whisper (overrides runtime.use_cached_asr)")
    parser.add_argument("--no-cache-embeddings", action="store_true", help="Ignore cached embeddings and re-run ECAPA (overrides runtime.use_cached_embeddings)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_experiment(
            config_path=args.config,
            output_dir=args.output_dir,
            meeting_id=args.meeting_id,
            dry_run=args.dry_run,
            fail_fast=args.fail_fast,
            use_cached_asr=not args.no_cache_asr if args.no_cache_asr else None,
            use_cached_embeddings=not args.no_cache_embeddings if args.no_cache_embeddings else None,
        )
        print(f"\nExperiment complete: {result['experiment_id']}")
        print(f"Output directory: {result['output_dir']}")
        summary = result.get("summary", {})
        if "num_completed" in summary:
            print(f"Meetings: {summary['num_completed']}/{summary['num_meetings_total']} completed")
            print(f"Mean DER: {summary.get('der', {}).get('mean')}")
            print(f"Mean WER: {summary.get('wer_integrated', {}).get('mean')}")
        return 0
    except Exception as exc:
        logger.error("Experiment failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
