"""Meeting audio reconstruction from HuggingFace dataset segments.

The AMI dataset uses pre-segmented utterances (one audio clip per row) in
recent ``datasets`` versions (4.8+). This module reconstructs the full meeting
audio by grouping segments by microphone channel, sorting by time, and placing
each clip at its correct offset in a single WAV file.
"""

import logging
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio

from src.data.ami_loader import AMILoader

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


def get_meeting_audio_path(
    loader: AMILoader,
    meeting_id: str,
    split: str,
    raw_dir: Path | None = None,
) -> str:
    """Resolve audio path, materialising full meeting audio from segments.

    Args:
        loader: AMILoader instance.
        meeting_id: AMI meeting identifier.
        split: Split to use (train/validation/test).
        raw_dir: Directory to cache reconstructed WAV files. Defaults to
            ``PROJECT_ROOT / data / raw / {config}``.

    Returns:
        Path to the reconstructed mono WAV file.
    """
    import sys

    # Resolve project root
    if "src" in __file__:
        # Running from project root
        project_root = Path(__file__).resolve().parents[2]
    else:
        # Fallback: try to find project root from CWD
        project_root = Path.cwd()

    if raw_dir is None:
        raw_dir = project_root / "data" / "raw" / loader.config
    raw_dir.mkdir(parents=True, exist_ok=True)
    wav_path = raw_dir / f"{meeting_id}.wav"
    if wav_path.exists():
        return str(wav_path)

    logger.info(
        "Reconstructing meeting audio for %s (%s/%s)…",
        meeting_id,
        loader.config,
        split,
    )

    # Load segments from the dataset
    segments = loader.get_meeting_segments(meeting_id, split)
    if not segments:
        raise FileNotFoundError(
            f"No segments found for {meeting_id} in AMI {loader.config}/{split}. "
            f"Ensure the HuggingFace dataset is downloaded."
        )

    # Group segments by microphone channel and sort by time
    channels: dict[str, list[dict]] = defaultdict(list)
    for seg in segments:
        channels[seg["microphone_id"]].append(seg)
    for ch in channels:
        channels[ch].sort(key=lambda s: s["begin_time"])

    # Determine full meeting duration
    max_end = max(seg["end_time"] for seg in segments)
    total_samples = int(max_end * TARGET_SR) + 1

    # Build multi-channel audio (one channel per microphone)
    channel_ids = sorted(channels.keys())
    num_channels = len(channel_ids)
    full_audio = torch.zeros(num_channels, total_samples, dtype=torch.float32)

    for ch_idx, ch_id in enumerate(channel_ids):
        for seg in channels[ch_id]:
            # Decode audio clip from the AudioDecoder
            audio_decoder = seg["audio"]
            samples = audio_decoder.get_all_samples()
            clip = samples.data  # shape (1, clip_samples)
            clip_sr = getattr(samples, "sample_rate", TARGET_SR)

            # Resample if needed (should be 16kHz already)
            if clip_sr != TARGET_SR:
                clip = torchaudio.functional.resample(clip, clip_sr, TARGET_SR)

            # Place at correct offset
            start_sample = int(seg["begin_time"] * TARGET_SR)
            clip_len = clip.shape[1]
            end_sample = min(start_sample + clip_len, total_samples)
            actual_len = end_sample - start_sample
            if actual_len > 0:
                full_audio[ch_idx, start_sample:end_sample] = clip[0, :actual_len]

    # Convert to mono for downstream compatibility (Whisper/ECAPA expect mono)
    if num_channels > 1:
        mono_audio = full_audio.mean(dim=0, keepdim=True)
    else:
        mono_audio = full_audio

    torchaudio.save(str(wav_path), mono_audio, TARGET_SR)
    logger.info(
        "Saved reconstructed audio: %s (%.1f s, %d Hz, mono)",
        wav_path,
        mono_audio.shape[1] / TARGET_SR,
        TARGET_SR,
    )
    return str(wav_path)
